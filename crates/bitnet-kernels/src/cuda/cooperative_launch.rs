//! Cooperative kernel launch configuration and grid-level synchronization.
//!
//! This module provides CPU-simulated cooperative launch primitives that mirror
//! CUDA's `cudaLaunchCooperativeKernel` and related APIs. On GPU these map to
//! hardware-accelerated grid-wide synchronization, multi-grid cooperative
//! launches, and block cluster support. The CPU fallback performs equivalent
//! sequential simulation for correctness testing and non-GPU environments.
//!
//! # Core types
//!
//! - [`CooperativeLaunchConfig`] — launch dimensions and cooperative flags
//! - [`GridSyncBarrier`] — grid-level synchronization barrier
//! - [`OccupancyConfig`] — occupancy-based launch configuration
//! - [`BlockClusterConfig`] — SM 9.0+ block cluster support (Hopper)
//! - [`MultiGridLaunchConfig`] — multi-GPU cooperative launch
//! - [`DeviceCooperativeCapabilities`] — capability detection
//!
//! # Functions
//!
//! - [`compute_cooperative_launch_config`] — occupancy-optimal grid dimensions
//! - [`grid_stride_loop`] — grid-stride loop pattern for arbitrary sizes
//! - [`grid_stride_loop_2d`] — 2D grid-stride loop
//! - [`launch_cooperative_kernel`] — simulate cooperative kernel launch
//! - [`launch_multi_grid_cooperative`] — multi-GPU cooperative launch
//! - [`query_cooperative_capabilities`] — device capability detection
//! - [`max_active_blocks_per_sm`] — occupancy query
//! - [`compute_cluster_launch_config`] — block cluster launch (SM 9.0+)
//!
//! # CUDA kernel source
//!
//! [`COOPERATIVE_LAUNCH_KERNEL_SRC`] contains CUDA C kernels that use
//! cooperative launch APIs. Feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source — cooperative launch
// ---------------------------------------------------------------------------

/// CUDA C kernel source implementing cooperative launch patterns.
///
/// Contains kernels for grid-wide synchronization, grid-stride loops,
/// multi-block reductions, and block cluster operations using
/// `cudaLaunchCooperativeKernel` and `cooperative_groups.h`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const COOPERATIVE_LAUNCH_KERNEL_SRC: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Grid-wide cooperative reduction using grid.sync().
// Requires cooperative launch (cudaLaunchCooperativeKernel).
extern "C" __global__ void coop_launch_grid_reduce_f32(
    float* __restrict__ data,
    float* __restrict__ output,
    int n)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Phase 1: block-local reduction
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) data[blockIdx.x] = sdata[0];

    // Grid-wide barrier
    grid.sync();

    // Phase 2: final reduction by block 0
    if (blockIdx.x == 0) {
        int num_blocks = gridDim.x;
        sdata[tid] = (tid < num_blocks) ? data[tid] : 0.0f;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) output[0] = sdata[0];
    }
}

// Grid-stride loop kernel for processing arbitrary data sizes.
extern "C" __global__ void coop_launch_grid_stride_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * scale;
    }
}

// Multi-phase cooperative kernel with grid barriers between phases.
extern "C" __global__ void coop_launch_multi_phase_f32(
    float* __restrict__ data,
    int n,
    int num_phases)
{
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int phase = 0; phase < num_phases; phase++) {
        for (int i = idx; i < n; i += stride) {
            data[i] = data[i] * 0.5f + 1.0f;
        }
        grid.sync();
    }
}

// Block cluster distributed shared memory (SM 9.0+ / Hopper).
#if __CUDA_ARCH__ >= 900
extern "C" __global__ void coop_launch_cluster_reduce_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n)
{
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    block.sync();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        block.sync();
    }

    // Cluster-level barrier before cross-block reduction
    cluster.sync();

    if (tid == 0 && cluster.block_rank() == 0) {
        float total = 0.0f;
        for (int b = 0; b < cluster.num_blocks(); b++) {
            float* remote_smem = cluster.map_shared_memory(sdata, b);
            total += remote_smem[0];
        }
        output[blockIdx.x / cluster.num_blocks()] = total;
    }
}
#endif
"#;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

static NEXT_BARRIER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_LAUNCH_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a grid sync barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BarrierId(u64);

impl BarrierId {
    fn next() -> Self {
        Self(NEXT_BARRIER_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for BarrierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "barrier-{}", self.0)
    }
}

/// Unique identifier for a cooperative launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaunchId(u64);

impl LaunchId {
    fn next() -> Self {
        Self(NEXT_LAUNCH_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for LaunchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "coop-launch-{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Compute capability
// ---------------------------------------------------------------------------

/// CUDA compute capability version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    /// Major version (e.g. 7 for Volta, 8 for Ampere, 9 for Hopper).
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a new compute capability.
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// SM 6.0+ (Pascal) — minimum for cooperative launch.
    pub fn supports_cooperative_launch(self) -> bool {
        self.major >= 6
    }

    /// SM 7.0+ (Volta) — supports multi-grid cooperative launch.
    pub fn supports_multi_grid(self) -> bool {
        self.major >= 7
    }

    /// SM 9.0+ (Hopper) — supports block clusters.
    pub fn supports_block_clusters(self) -> bool {
        self.major >= 9
    }

    /// Numeric representation (e.g. 75 for SM 7.5).
    pub fn as_numeric(self) -> u32 {
        self.major * 10 + self.minor
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM {}.{}", self.major, self.minor)
    }
}

// ---------------------------------------------------------------------------
// Device cooperative capabilities
// ---------------------------------------------------------------------------

/// Device capabilities for cooperative kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCooperativeCapabilities {
    /// Device index.
    pub device_id: u32,
    /// Compute capability.
    pub compute_capability: ComputeCapability,
    /// Number of streaming multiprocessors on the device.
    pub sm_count: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Maximum threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum shared memory per block in bytes.
    pub max_shared_mem_per_block: u32,
    /// Maximum shared memory per SM in bytes.
    pub max_shared_mem_per_sm: u32,
    /// Maximum number of active blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Whether cooperative launch is supported.
    pub cooperative_launch: bool,
    /// Whether multi-device cooperative launch is supported.
    pub cooperative_multi_device_launch: bool,
    /// Whether block cluster launch is supported (SM 9.0+).
    pub block_cluster_launch: bool,
}

impl Default for DeviceCooperativeCapabilities {
    fn default() -> Self {
        Self {
            device_id: 0,
            compute_capability: ComputeCapability::new(8, 0),
            sm_count: 108,
            max_threads_per_block: 1024,
            max_threads_per_sm: 2048,
            max_shared_mem_per_block: 48 * 1024,
            max_shared_mem_per_sm: 164 * 1024,
            max_blocks_per_sm: 32,
            cooperative_launch: true,
            cooperative_multi_device_launch: true,
            block_cluster_launch: false,
        }
    }
}

impl DeviceCooperativeCapabilities {
    /// Create capabilities for a Hopper (SM 9.0) device.
    pub fn hopper() -> Self {
        Self {
            compute_capability: ComputeCapability::new(9, 0),
            sm_count: 132,
            max_threads_per_sm: 2048,
            max_shared_mem_per_block: 228 * 1024,
            max_shared_mem_per_sm: 228 * 1024,
            max_blocks_per_sm: 32,
            block_cluster_launch: true,
            ..Default::default()
        }
    }

    /// Create capabilities for an Ampere (SM 8.0) device.
    pub fn ampere() -> Self {
        Self::default()
    }

    /// Create capabilities for a Volta (SM 7.0) device.
    pub fn volta() -> Self {
        Self {
            compute_capability: ComputeCapability::new(7, 0),
            sm_count: 80,
            max_threads_per_sm: 2048,
            max_shared_mem_per_block: 96 * 1024,
            max_shared_mem_per_sm: 96 * 1024,
            max_blocks_per_sm: 32,
            ..Default::default()
        }
    }

    /// Maximum total concurrent threads on the device.
    pub fn max_concurrent_threads(&self) -> u64 {
        u64::from(self.sm_count) * u64::from(self.max_threads_per_sm)
    }
}

// ---------------------------------------------------------------------------
// Query cooperative capabilities (CPU fallback)
// ---------------------------------------------------------------------------

/// Query device cooperative launch capabilities.
///
/// On CPU this returns a simulated default device (Ampere SM 8.0).
/// On GPU this would call `cudaDeviceGetAttribute`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `device_id` exceeds 255.
pub fn query_cooperative_capabilities(device_id: u32) -> Result<DeviceCooperativeCapabilities> {
    if device_id > 255 {
        return Err(KernelError::InvalidArguments {
            reason: format!("device_id must be 0..=255, got {device_id}"),
        }
        .into());
    }
    Ok(DeviceCooperativeCapabilities { device_id, ..Default::default() })
}

// ---------------------------------------------------------------------------
// Cooperative launch config
// ---------------------------------------------------------------------------

/// Configuration for a cooperative kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooperativeLaunchConfig {
    /// Grid dimensions (blocks in x, y, z).
    pub grid: [u32; 3],
    /// Block dimensions (threads in x, y, z).
    pub block: [u32; 3],
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// Stream index (0 = default).
    pub stream: u32,
    /// Enable grid-wide synchronization (cooperative launch).
    pub cooperative: bool,
    /// Enable multi-device cooperative launch.
    pub multi_device: bool,
}

impl Default for CooperativeLaunchConfig {
    fn default() -> Self {
        Self {
            grid: [1, 1, 1],
            block: [256, 1, 1],
            shared_mem_bytes: 0,
            stream: 0,
            cooperative: true,
            multi_device: false,
        }
    }
}

impl CooperativeLaunchConfig {
    /// Create a 1D cooperative launch config.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if dimensions are invalid.
    pub fn new_1d(grid_x: u32, block_x: u32) -> Result<Self> {
        if grid_x == 0 || block_x == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "grid and block dimensions must be non-zero: grid_x={grid_x}, block_x={block_x}"
                ),
            }
            .into());
        }
        if block_x > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: format!("block_x must be <= 1024, got {block_x}"),
            }
            .into());
        }
        Ok(Self { grid: [grid_x, 1, 1], block: [block_x, 1, 1], ..Default::default() })
    }

    /// Create a 2D cooperative launch config.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if dimensions are invalid.
    pub fn new_2d(grid: [u32; 2], block: [u32; 2]) -> Result<Self> {
        if grid[0] == 0 || grid[1] == 0 || block[0] == 0 || block[1] == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "all grid and block dimensions must be non-zero".into(),
            }
            .into());
        }
        let threads_per_block = block[0] as u64 * block[1] as u64;
        if threads_per_block > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: format!("threads per block must be <= 1024, got {threads_per_block}"),
            }
            .into());
        }
        Ok(Self {
            grid: [grid[0], grid[1], 1],
            block: [block[0], block[1], 1],
            ..Default::default()
        })
    }

    /// Create a 3D cooperative launch config.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if dimensions are invalid.
    pub fn new_3d(grid: [u32; 3], block: [u32; 3]) -> Result<Self> {
        for (i, &g) in grid.iter().enumerate() {
            if g == 0 {
                return Err(KernelError::InvalidArguments {
                    reason: format!("grid[{i}] must be non-zero"),
                }
                .into());
            }
        }
        for (i, &b) in block.iter().enumerate() {
            if b == 0 {
                return Err(KernelError::InvalidArguments {
                    reason: format!("block[{i}] must be non-zero"),
                }
                .into());
            }
        }
        let threads = block[0] as u64 * block[1] as u64 * block[2] as u64;
        if threads > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: format!("threads per block must be <= 1024, got {threads}"),
            }
            .into());
        }
        Ok(Self { grid, block, ..Default::default() })
    }

    /// Set shared memory size.
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Set stream index.
    #[must_use]
    pub fn with_stream(mut self, stream: u32) -> Self {
        self.stream = stream;
        self
    }

    /// Disable cooperative mode (standard launch).
    #[must_use]
    pub fn non_cooperative(mut self) -> Self {
        self.cooperative = false;
        self
    }

    /// Enable multi-device cooperative launch.
    #[must_use]
    pub fn with_multi_device(mut self) -> Self {
        self.multi_device = true;
        self
    }

    /// Total number of threads in the grid.
    pub fn total_threads(&self) -> u64 {
        let grid_total = self.grid[0] as u64 * self.grid[1] as u64 * self.grid[2] as u64;
        let block_total = self.block[0] as u64 * self.block[1] as u64 * self.block[2] as u64;
        grid_total * block_total
    }

    /// Total number of blocks in the grid.
    pub fn total_blocks(&self) -> u64 {
        self.grid[0] as u64 * self.grid[1] as u64 * self.grid[2] as u64
    }

    /// Threads per block.
    pub fn threads_per_block(&self) -> u32 {
        self.block[0] * self.block[1] * self.block[2]
    }

    /// Validate config against device capabilities.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if the config exceeds device limits.
    pub fn validate(&self, caps: &DeviceCooperativeCapabilities) -> Result<()> {
        if self.cooperative && !caps.cooperative_launch {
            return Err(KernelError::InvalidArguments {
                reason: format!("device {} does not support cooperative launch", caps.device_id),
            }
            .into());
        }
        if self.multi_device && !caps.cooperative_multi_device_launch {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "device {} does not support multi-device cooperative launch",
                    caps.device_id
                ),
            }
            .into());
        }
        let tpb = self.threads_per_block();
        if tpb > caps.max_threads_per_block {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "threads per block ({tpb}) exceeds device max ({})",
                    caps.max_threads_per_block
                ),
            }
            .into());
        }
        if self.shared_mem_bytes > caps.max_shared_mem_per_block {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "shared memory ({} bytes) exceeds device max ({} bytes)",
                    self.shared_mem_bytes, caps.max_shared_mem_per_block
                ),
            }
            .into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Grid sync barrier
// ---------------------------------------------------------------------------

/// Grid-level synchronization barrier state.
///
/// Tracks barrier arrivals across all blocks. On GPU this maps to
/// `cooperative_groups::grid_group::sync()`.
#[derive(Debug, Clone)]
pub struct GridSyncBarrier {
    /// Unique barrier identifier.
    pub id: BarrierId,
    /// Total number of blocks that must arrive.
    pub num_blocks: u32,
    /// Number of blocks that have arrived.
    arrived: u32,
    /// Number of times this barrier has been completed.
    pub generation: u64,
}

impl GridSyncBarrier {
    /// Create a new grid sync barrier for the given number of blocks.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `num_blocks` is 0.
    pub fn new(num_blocks: u32) -> Result<Self> {
        if num_blocks == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "barrier num_blocks must be > 0".into(),
            }
            .into());
        }
        Ok(Self { id: BarrierId::next(), num_blocks, arrived: 0, generation: 0 })
    }

    /// Record a block arriving at the barrier. Returns `true` if all blocks
    /// have arrived (barrier is complete).
    pub fn arrive(&mut self) -> bool {
        self.arrived += 1;
        if self.arrived >= self.num_blocks {
            self.arrived = 0;
            self.generation += 1;
            true
        } else {
            false
        }
    }

    /// Reset the barrier for a new synchronization round.
    pub fn reset(&mut self) {
        self.arrived = 0;
    }

    /// Number of blocks still pending.
    pub fn pending(&self) -> u32 {
        self.num_blocks - self.arrived
    }

    /// Whether the barrier is currently complete (no pending arrivals).
    pub fn is_complete(&self) -> bool {
        self.arrived == 0 && self.generation > 0
    }
}

// ---------------------------------------------------------------------------
// Occupancy config
// ---------------------------------------------------------------------------

/// Occupancy-based launch configuration inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OccupancyConfig {
    /// Threads per block for occupancy calculation.
    pub block_size: u32,
    /// Dynamic shared memory per block in bytes.
    pub dynamic_shared_mem: u32,
    /// Number of registers per thread (0 = auto).
    pub registers_per_thread: u32,
}

impl Default for OccupancyConfig {
    fn default() -> Self {
        Self { block_size: 256, dynamic_shared_mem: 0, registers_per_thread: 0 }
    }
}

impl OccupancyConfig {
    /// Create with the given block size.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `block_size` is 0 or > 1024.
    pub fn new(block_size: u32) -> Result<Self> {
        if block_size == 0 || block_size > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: format!("block_size must be 1..=1024, got {block_size}"),
            }
            .into());
        }
        Ok(Self { block_size, ..Default::default() })
    }

    /// Set dynamic shared memory.
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.dynamic_shared_mem = bytes;
        self
    }

    /// Set registers per thread.
    #[must_use]
    pub fn with_registers(mut self, regs: u32) -> Self {
        self.registers_per_thread = regs;
        self
    }
}

/// Result of an occupancy calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OccupancyResult {
    /// Maximum active blocks per SM.
    pub max_active_blocks_per_sm: u32,
    /// Active warps per SM.
    pub active_warps_per_sm: u32,
    /// Maximum warps per SM.
    pub max_warps_per_sm: u32,
    /// Occupancy percentage (0–100).
    pub occupancy_percent: u32,
    /// Recommended grid size for full device occupancy.
    pub recommended_grid_size: u32,
}

// ---------------------------------------------------------------------------
// Occupancy calculation (CPU simulation)
// ---------------------------------------------------------------------------

/// Calculate maximum active blocks per SM for a cooperative kernel.
///
/// Simulates `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Considers
/// thread count, shared memory, and register limits.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] for invalid configurations.
pub fn max_active_blocks_per_sm(
    config: &OccupancyConfig,
    caps: &DeviceCooperativeCapabilities,
) -> Result<OccupancyResult> {
    if config.block_size == 0 || config.block_size > caps.max_threads_per_block {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "block_size ({}) must be 1..={}",
                config.block_size, caps.max_threads_per_block
            ),
        }
        .into());
    }

    // Thread limit
    let thread_limited = caps.max_threads_per_sm / config.block_size;

    // Shared memory limit
    let smem_per_block = if config.dynamic_shared_mem > 0 {
        config.dynamic_shared_mem
    } else {
        // Assume a minimal static shared memory allocation
        1024
    };
    let smem_limited =
        if smem_per_block > 0 { caps.max_shared_mem_per_sm / smem_per_block } else { u32::MAX };

    // Register limit (simplified model: 65536 registers per SM)
    let regs_per_sm = 65536u32;
    let regs_per_thread = if config.registers_per_thread > 0 {
        config.registers_per_thread
    } else {
        32 // reasonable default
    };
    let warps_per_block = config.block_size.div_ceil(32);
    let regs_per_block = warps_per_block * 32 * regs_per_thread;
    let reg_limited = if regs_per_block > 0 { regs_per_sm / regs_per_block } else { u32::MAX };

    let max_blocks = thread_limited.min(smem_limited).min(reg_limited).min(caps.max_blocks_per_sm);

    let active_warps = max_blocks * warps_per_block;
    let max_warps = caps.max_threads_per_sm / 32;
    let occupancy_pct = if max_warps > 0 { ((active_warps * 100) / max_warps).min(100) } else { 0 };

    let recommended_grid = max_blocks * caps.sm_count;

    Ok(OccupancyResult {
        max_active_blocks_per_sm: max_blocks,
        active_warps_per_sm: active_warps,
        max_warps_per_sm: max_warps,
        occupancy_percent: occupancy_pct,
        recommended_grid_size: recommended_grid,
    })
}

/// Compute cooperative launch configuration for a given data size.
///
/// Determines optimal grid/block dimensions for a cooperative kernel,
/// clamped to the device's maximum concurrent block count.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data_size` is 0.
pub fn compute_cooperative_launch_config(
    data_size: usize,
    occupancy: &OccupancyConfig,
    caps: &DeviceCooperativeCapabilities,
) -> Result<CooperativeLaunchConfig> {
    if data_size == 0 {
        return Err(KernelError::InvalidArguments { reason: "data_size must be > 0".into() }.into());
    }

    let occ = max_active_blocks_per_sm(occupancy, caps)?;

    // Number of blocks needed to cover data
    let blocks_needed = (data_size as u64).div_ceil(occupancy.block_size as u64) as u32;

    // Clamp to max cooperative grid size
    let max_coop_blocks = occ.max_active_blocks_per_sm * caps.sm_count;
    let grid_x = blocks_needed.min(max_coop_blocks).max(1);

    let mut config = CooperativeLaunchConfig::new_1d(grid_x, occupancy.block_size)?;
    config.shared_mem_bytes = occupancy.dynamic_shared_mem;
    Ok(config)
}

// ---------------------------------------------------------------------------
// Block cluster config (SM 9.0+ / Hopper)
// ---------------------------------------------------------------------------

/// Configuration for block cluster launch (SM 9.0+, Hopper architecture).
///
/// Block clusters enable distributed shared memory across neighbouring
/// thread blocks scheduled on the same GPC (Graphics Processing Cluster).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockClusterConfig {
    /// Number of blocks per cluster (1–16, must be power of 2).
    pub cluster_size: u32,
    /// Whether to use distributed shared memory.
    pub distributed_shared_mem: bool,
    /// Total distributed shared memory across the cluster in bytes.
    pub total_cluster_shared_mem: u32,
    /// Launch dimensions (blocks within the cluster grid).
    pub cluster_grid: [u32; 3],
}

impl Default for BlockClusterConfig {
    fn default() -> Self {
        Self {
            cluster_size: 2,
            distributed_shared_mem: false,
            total_cluster_shared_mem: 0,
            cluster_grid: [1, 1, 1],
        }
    }
}

impl BlockClusterConfig {
    /// Create a cluster config with the given size.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `size` is 0, > 16, or
    /// not a power of 2.
    pub fn new(size: u32) -> Result<Self> {
        if size == 0 || size > 16 {
            return Err(KernelError::InvalidArguments {
                reason: format!("cluster_size must be 1..=16, got {size}"),
            }
            .into());
        }
        if !size.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: format!("cluster_size must be a power of 2, got {size}"),
            }
            .into());
        }
        Ok(Self { cluster_size: size, ..Default::default() })
    }

    /// Enable distributed shared memory.
    #[must_use]
    pub fn with_distributed_shared_mem(mut self, bytes: u32) -> Self {
        self.distributed_shared_mem = true;
        self.total_cluster_shared_mem = bytes;
        self
    }

    /// Set cluster grid dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any dimension is 0.
    pub fn with_cluster_grid(mut self, grid: [u32; 3]) -> Result<Self> {
        for (i, &g) in grid.iter().enumerate() {
            if g == 0 {
                return Err(KernelError::InvalidArguments {
                    reason: format!("cluster_grid[{i}] must be non-zero"),
                }
                .into());
            }
        }
        self.cluster_grid = grid;
        Ok(self)
    }

    /// Total blocks in the cluster grid.
    pub fn total_cluster_blocks(&self) -> u64 {
        self.cluster_grid[0] as u64
            * self.cluster_grid[1] as u64
            * self.cluster_grid[2] as u64
            * self.cluster_size as u64
    }

    /// Validate against device capabilities.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if the device doesn't support clusters.
    pub fn validate(&self, caps: &DeviceCooperativeCapabilities) -> Result<()> {
        if !caps.block_cluster_launch {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "device {} ({}) does not support block clusters (requires SM 9.0+)",
                    caps.device_id, caps.compute_capability
                ),
            }
            .into());
        }
        Ok(())
    }
}

/// Compute cluster launch config for a given data size and cluster parameters.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] for invalid parameters or
/// unsupported devices.
pub fn compute_cluster_launch_config(
    data_size: usize,
    block_size: u32,
    cluster: &BlockClusterConfig,
    caps: &DeviceCooperativeCapabilities,
) -> Result<CooperativeLaunchConfig> {
    cluster.validate(caps)?;

    if data_size == 0 {
        return Err(KernelError::InvalidArguments { reason: "data_size must be > 0".into() }.into());
    }
    if block_size == 0 || block_size > 1024 {
        return Err(KernelError::InvalidArguments {
            reason: format!("block_size must be 1..=1024, got {block_size}"),
        }
        .into());
    }

    let threads_per_cluster = block_size as u64 * cluster.cluster_size as u64;
    let clusters_needed = (data_size as u64).div_ceil(threads_per_cluster) as u32;
    let total_blocks = clusters_needed * cluster.cluster_size;

    let config = CooperativeLaunchConfig {
        grid: [total_blocks, 1, 1],
        block: [block_size, 1, 1],
        shared_mem_bytes: cluster.total_cluster_shared_mem / cluster.cluster_size,
        stream: 0,
        cooperative: true,
        multi_device: false,
    };
    Ok(config)
}

// ---------------------------------------------------------------------------
// Multi-grid cooperative launch
// ---------------------------------------------------------------------------

/// Configuration for multi-GPU cooperative kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiGridLaunchConfig {
    /// Per-device launch configs (one per GPU).
    pub device_configs: Vec<DeviceLaunchEntry>,
    /// Synchronization mode between devices.
    pub sync_mode: MultiGridSyncMode,
}

/// Per-device entry in a multi-grid launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLaunchEntry {
    /// Device index.
    pub device_id: u32,
    /// Launch config for this device.
    pub config: CooperativeLaunchConfig,
    /// Data range (start element, count).
    pub data_range: (usize, usize),
}

/// Synchronization mode for multi-grid launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiGridSyncMode {
    /// All devices synchronize at grid barriers.
    FullSync,
    /// Only neighbouring devices synchronize (pipeline).
    PipelineSync,
    /// No inter-device synchronization (independent).
    Independent,
}

impl MultiGridLaunchConfig {
    /// Create a multi-grid config distributing `data_size` across devices.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if no device capabilities
    /// are provided or data_size is 0.
    pub fn new(
        data_size: usize,
        block_size: u32,
        device_caps: &[DeviceCooperativeCapabilities],
        sync_mode: MultiGridSyncMode,
    ) -> Result<Self> {
        if device_caps.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "at least one device is required".into(),
            }
            .into());
        }
        if data_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "data_size must be > 0".into() }.into()
            );
        }
        if block_size == 0 || block_size > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: format!("block_size must be 1..=1024, got {block_size}"),
            }
            .into());
        }

        let num_devices = device_caps.len();
        let chunk_size = data_size.div_ceil(num_devices);
        let mut entries = Vec::with_capacity(num_devices);

        for (i, caps) in device_caps.iter().enumerate() {
            let start = i * chunk_size;
            let count = chunk_size.min(data_size.saturating_sub(start));
            if count == 0 {
                continue;
            }
            let grid_x = (count as u64).div_ceil(block_size as u64) as u32;
            let config = CooperativeLaunchConfig {
                grid: [grid_x.max(1), 1, 1],
                block: [block_size, 1, 1],
                cooperative: true,
                multi_device: true,
                ..Default::default()
            };
            entries.push(DeviceLaunchEntry {
                device_id: caps.device_id,
                config,
                data_range: (start, count),
            });
        }

        Ok(Self { device_configs: entries, sync_mode })
    }

    /// Total number of participating devices.
    pub fn num_devices(&self) -> usize {
        self.device_configs.len()
    }

    /// Total threads across all devices.
    pub fn total_threads(&self) -> u64 {
        self.device_configs.iter().map(|e| e.config.total_threads()).sum()
    }
}

// ---------------------------------------------------------------------------
// Grid-stride loop patterns (CPU simulation)
// ---------------------------------------------------------------------------

/// Execute a grid-stride loop over `data`, applying `func` to each element.
///
/// Simulates the CUDA pattern:
/// ```text
/// int idx = blockIdx.x * blockDim.x + threadIdx.x;
/// int stride = gridDim.x * blockDim.x;
/// for (int i = idx; i < n; i += stride) { ... }
/// ```
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty.
pub fn grid_stride_loop(
    data: &mut [f32],
    config: &CooperativeLaunchConfig,
    func: impl Fn(f32, usize) -> f32,
) -> Result<()> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "grid_stride_loop data must not be empty".into(),
        }
        .into());
    }

    let total_threads = config.total_threads() as usize;
    let stride = total_threads.max(1);

    for thread_id in 0..stride.min(data.len()) {
        let mut i = thread_id;
        while i < data.len() {
            data[i] = func(data[i], i);
            i += stride;
        }
    }

    Ok(())
}

/// 2D grid-stride loop over a matrix stored in row-major order.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if dimensions are invalid.
pub fn grid_stride_loop_2d(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    config: &CooperativeLaunchConfig,
    func: impl Fn(f32, usize, usize) -> f32,
) -> Result<()> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "grid_stride_loop_2d data must not be empty".into(),
        }
        .into());
    }
    if rows * cols != data.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("rows*cols ({}) != data.len() ({})", rows * cols, data.len()),
        }
        .into());
    }

    let grid_x = config.grid[0] as usize;
    let grid_y = config.grid[1] as usize;
    let block_x = config.block[0] as usize;
    let block_y = config.block[1] as usize;

    let stride_x = grid_x * block_x;
    let stride_y = grid_y * block_y;

    for by in 0..grid_y {
        for bx in 0..grid_x {
            for ty in 0..block_y {
                for tx in 0..block_x {
                    let start_row = by * block_y + ty;
                    let start_col = bx * block_x + tx;
                    let mut r = start_row;
                    while r < rows {
                        let mut c = start_col;
                        while c < cols {
                            let idx = r * cols + c;
                            data[idx] = func(data[idx], r, c);
                            c += stride_x;
                        }
                        r += stride_y;
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cooperative kernel launch (CPU simulation)
// ---------------------------------------------------------------------------

/// Result of a cooperative kernel launch.
#[derive(Debug, Clone)]
pub struct CooperativeLaunchResult {
    /// Launch identifier.
    pub id: LaunchId,
    /// Number of blocks executed.
    pub blocks_executed: u64,
    /// Total elements processed.
    pub elements_processed: usize,
    /// Whether grid sync was used.
    pub grid_sync_used: bool,
}

/// Launch a cooperative kernel (CPU simulation).
///
/// Simulates `cudaLaunchCooperativeKernel` by executing the kernel function
/// sequentially with grid-stride looping. The `kernel_fn` receives the data
/// slice and a grid sync barrier for multi-phase kernels.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty.
pub fn launch_cooperative_kernel(
    data: &mut [f32],
    config: &CooperativeLaunchConfig,
    kernel_fn: impl Fn(&mut [f32], &mut GridSyncBarrier),
) -> Result<CooperativeLaunchResult> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "cooperative kernel data must not be empty".into(),
        }
        .into());
    }

    let id = LaunchId::next();
    let num_blocks = config.total_blocks() as u32;
    let mut barrier = GridSyncBarrier::new(num_blocks)?;

    kernel_fn(data, &mut barrier);

    // Simulate all blocks arriving
    for _ in 0..num_blocks {
        barrier.arrive();
    }

    Ok(CooperativeLaunchResult {
        id,
        blocks_executed: config.total_blocks(),
        elements_processed: data.len(),
        grid_sync_used: config.cooperative,
    })
}

/// Launch a multi-grid cooperative kernel across devices (CPU simulation).
///
/// Simulates `cudaLaunchCooperativeKernelMultiDevice`. Processes each device's
/// data range sequentially.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty or
/// any device range is out of bounds.
pub fn launch_multi_grid_cooperative(
    data: &mut [f32],
    multi_config: &MultiGridLaunchConfig,
    kernel_fn: impl Fn(&mut [f32], u32),
) -> Result<Vec<CooperativeLaunchResult>> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "multi-grid data must not be empty".into(),
        }
        .into());
    }

    let mut results = Vec::with_capacity(multi_config.num_devices());

    for entry in &multi_config.device_configs {
        let (start, count) = entry.data_range;
        let end = start + count;
        if end > data.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "device {} range [{start}..{end}) exceeds data len {}",
                    entry.device_id,
                    data.len()
                ),
            }
            .into());
        }
        kernel_fn(&mut data[start..end], entry.device_id);
        results.push(CooperativeLaunchResult {
            id: LaunchId::next(),
            blocks_executed: entry.config.total_blocks(),
            elements_processed: count,
            grid_sync_used: entry.config.cooperative,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_caps() -> DeviceCooperativeCapabilities {
        DeviceCooperativeCapabilities::default()
    }

    fn hopper_caps() -> DeviceCooperativeCapabilities {
        DeviceCooperativeCapabilities::hopper()
    }

    // ===== ComputeCapability =====

    #[test]
    fn test_compute_capability_cooperative_launch_sm60() {
        let cc = ComputeCapability::new(6, 0);
        assert!(cc.supports_cooperative_launch());
        assert!(!cc.supports_multi_grid());
        assert!(!cc.supports_block_clusters());
    }

    #[test]
    fn test_compute_capability_volta_sm70() {
        let cc = ComputeCapability::new(7, 0);
        assert!(cc.supports_cooperative_launch());
        assert!(cc.supports_multi_grid());
        assert!(!cc.supports_block_clusters());
    }

    #[test]
    fn test_compute_capability_hopper_sm90() {
        let cc = ComputeCapability::new(9, 0);
        assert!(cc.supports_cooperative_launch());
        assert!(cc.supports_multi_grid());
        assert!(cc.supports_block_clusters());
    }

    #[test]
    fn test_compute_capability_as_numeric() {
        assert_eq!(ComputeCapability::new(7, 5).as_numeric(), 75);
        assert_eq!(ComputeCapability::new(8, 6).as_numeric(), 86);
        assert_eq!(ComputeCapability::new(9, 0).as_numeric(), 90);
    }

    #[test]
    fn test_compute_capability_display() {
        assert_eq!(format!("{}", ComputeCapability::new(8, 0)), "SM 8.0");
    }

    #[test]
    fn test_compute_capability_ordering() {
        let v = ComputeCapability::new(7, 0);
        let a = ComputeCapability::new(8, 0);
        assert!(v < a);
    }

    #[test]
    fn test_compute_capability_equality() {
        let a = ComputeCapability::new(8, 0);
        let b = ComputeCapability::new(8, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_compute_capability_sm75() {
        let cc = ComputeCapability::new(7, 5);
        assert!(cc.supports_cooperative_launch());
        assert!(cc.supports_multi_grid());
        assert!(!cc.supports_block_clusters());
    }

    // ===== DeviceCooperativeCapabilities =====

    #[test]
    fn test_default_caps_ampere() {
        let caps = default_caps();
        assert_eq!(caps.compute_capability, ComputeCapability::new(8, 0));
        assert!(caps.cooperative_launch);
        assert!(caps.cooperative_multi_device_launch);
        assert!(!caps.block_cluster_launch);
        assert_eq!(caps.sm_count, 108);
    }

    #[test]
    fn test_hopper_caps() {
        let caps = hopper_caps();
        assert_eq!(caps.compute_capability, ComputeCapability::new(9, 0));
        assert!(caps.block_cluster_launch);
        assert_eq!(caps.sm_count, 132);
    }

    #[test]
    fn test_volta_caps() {
        let caps = DeviceCooperativeCapabilities::volta();
        assert_eq!(caps.compute_capability, ComputeCapability::new(7, 0));
        assert_eq!(caps.sm_count, 80);
    }

    #[test]
    fn test_max_concurrent_threads() {
        let caps = default_caps();
        assert_eq!(caps.max_concurrent_threads(), 108 * 2048);
    }

    // ===== query_cooperative_capabilities =====

    #[test]
    fn test_query_capabilities_valid() {
        let caps = query_cooperative_capabilities(0).unwrap();
        assert_eq!(caps.device_id, 0);
        assert!(caps.cooperative_launch);
    }

    #[test]
    fn test_query_capabilities_device_255() {
        let caps = query_cooperative_capabilities(255).unwrap();
        assert_eq!(caps.device_id, 255);
    }

    #[test]
    fn test_query_capabilities_invalid_device() {
        let err = query_cooperative_capabilities(256).unwrap_err();
        assert!(err.to_string().contains("device_id"));
    }

    // ===== CooperativeLaunchConfig =====

    #[test]
    fn test_config_new_1d() {
        let cfg = CooperativeLaunchConfig::new_1d(128, 256).unwrap();
        assert_eq!(cfg.grid, [128, 1, 1]);
        assert_eq!(cfg.block, [256, 1, 1]);
        assert!(cfg.cooperative);
        assert!(!cfg.multi_device);
    }

    #[test]
    fn test_config_new_1d_zero_grid() {
        assert!(CooperativeLaunchConfig::new_1d(0, 256).is_err());
    }

    #[test]
    fn test_config_new_1d_zero_block() {
        assert!(CooperativeLaunchConfig::new_1d(1, 0).is_err());
    }

    #[test]
    fn test_config_new_1d_block_too_large() {
        assert!(CooperativeLaunchConfig::new_1d(1, 2048).is_err());
    }

    #[test]
    fn test_config_new_2d() {
        let cfg = CooperativeLaunchConfig::new_2d([4, 8], [16, 16]).unwrap();
        assert_eq!(cfg.grid, [4, 8, 1]);
        assert_eq!(cfg.block, [16, 16, 1]);
    }

    #[test]
    fn test_config_new_2d_zero_dim() {
        assert!(CooperativeLaunchConfig::new_2d([0, 1], [16, 16]).is_err());
        assert!(CooperativeLaunchConfig::new_2d([1, 1], [0, 16]).is_err());
    }

    #[test]
    fn test_config_new_2d_too_many_threads() {
        assert!(CooperativeLaunchConfig::new_2d([1, 1], [64, 32]).is_err());
    }

    #[test]
    fn test_config_new_3d() {
        let cfg = CooperativeLaunchConfig::new_3d([2, 2, 2], [8, 8, 4]).unwrap();
        assert_eq!(cfg.grid, [2, 2, 2]);
        assert_eq!(cfg.block, [8, 8, 4]);
        assert_eq!(cfg.threads_per_block(), 256);
    }

    #[test]
    fn test_config_new_3d_zero_grid() {
        assert!(CooperativeLaunchConfig::new_3d([0, 1, 1], [8, 8, 4]).is_err());
    }

    #[test]
    fn test_config_new_3d_zero_block() {
        assert!(CooperativeLaunchConfig::new_3d([1, 1, 1], [8, 0, 4]).is_err());
    }

    #[test]
    fn test_config_new_3d_too_many_threads() {
        assert!(CooperativeLaunchConfig::new_3d([1, 1, 1], [32, 32, 2]).is_err());
    }

    #[test]
    fn test_config_total_threads() {
        let cfg = CooperativeLaunchConfig::new_1d(10, 256).unwrap();
        assert_eq!(cfg.total_threads(), 2560);
    }

    #[test]
    fn test_config_total_blocks() {
        let cfg = CooperativeLaunchConfig::new_2d([4, 8], [16, 16]).unwrap();
        assert_eq!(cfg.total_blocks(), 32);
    }

    #[test]
    fn test_config_threads_per_block() {
        let cfg = CooperativeLaunchConfig::new_1d(1, 512).unwrap();
        assert_eq!(cfg.threads_per_block(), 512);
    }

    #[test]
    fn test_config_with_shared_mem() {
        let cfg = CooperativeLaunchConfig::new_1d(1, 256).unwrap().with_shared_mem(4096);
        assert_eq!(cfg.shared_mem_bytes, 4096);
    }

    #[test]
    fn test_config_with_stream() {
        let cfg = CooperativeLaunchConfig::new_1d(1, 256).unwrap().with_stream(3);
        assert_eq!(cfg.stream, 3);
    }

    #[test]
    fn test_config_non_cooperative() {
        let cfg = CooperativeLaunchConfig::new_1d(1, 256).unwrap().non_cooperative();
        assert!(!cfg.cooperative);
    }

    #[test]
    fn test_config_with_multi_device() {
        let cfg = CooperativeLaunchConfig::new_1d(1, 256).unwrap().with_multi_device();
        assert!(cfg.multi_device);
    }

    #[test]
    fn test_config_default() {
        let cfg = CooperativeLaunchConfig::default();
        assert_eq!(cfg.grid, [1, 1, 1]);
        assert_eq!(cfg.block, [256, 1, 1]);
        assert!(cfg.cooperative);
    }

    #[test]
    fn test_config_validate_ok() {
        let caps = default_caps();
        let cfg = CooperativeLaunchConfig::new_1d(108, 256).unwrap();
        assert!(cfg.validate(&caps).is_ok());
    }

    #[test]
    fn test_config_validate_too_many_threads() {
        let mut caps = default_caps();
        caps.max_threads_per_block = 128;
        let cfg = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        assert!(cfg.validate(&caps).is_err());
    }

    #[test]
    fn test_config_validate_too_much_shared_mem() {
        let caps = default_caps();
        let cfg = CooperativeLaunchConfig::new_1d(1, 256)
            .unwrap()
            .with_shared_mem(caps.max_shared_mem_per_block + 1);
        assert!(cfg.validate(&caps).is_err());
    }

    #[test]
    fn test_config_validate_no_cooperative_support() {
        let mut caps = default_caps();
        caps.cooperative_launch = false;
        let cfg = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        assert!(cfg.validate(&caps).is_err());
    }

    #[test]
    fn test_config_validate_no_multi_device_support() {
        let mut caps = default_caps();
        caps.cooperative_multi_device_launch = false;
        let cfg = CooperativeLaunchConfig::new_1d(1, 256).unwrap().with_multi_device();
        assert!(cfg.validate(&caps).is_err());
    }

    // ===== GridSyncBarrier =====

    #[test]
    fn test_barrier_new() {
        let b = GridSyncBarrier::new(4).unwrap();
        assert_eq!(b.num_blocks, 4);
        assert_eq!(b.pending(), 4);
        assert_eq!(b.generation, 0);
    }

    #[test]
    fn test_barrier_new_zero_blocks() {
        assert!(GridSyncBarrier::new(0).is_err());
    }

    #[test]
    fn test_barrier_arrive_partial() {
        let mut b = GridSyncBarrier::new(4).unwrap();
        assert!(!b.arrive());
        assert!(!b.arrive());
        assert!(!b.arrive());
        assert_eq!(b.pending(), 1);
    }

    #[test]
    fn test_barrier_arrive_complete() {
        let mut b = GridSyncBarrier::new(3).unwrap();
        assert!(!b.arrive());
        assert!(!b.arrive());
        assert!(b.arrive());
        assert_eq!(b.generation, 1);
    }

    #[test]
    fn test_barrier_multiple_generations() {
        let mut b = GridSyncBarrier::new(2).unwrap();
        assert!(!b.arrive());
        assert!(b.arrive());
        assert_eq!(b.generation, 1);
        assert!(!b.arrive());
        assert!(b.arrive());
        assert_eq!(b.generation, 2);
    }

    #[test]
    fn test_barrier_reset() {
        let mut b = GridSyncBarrier::new(4).unwrap();
        b.arrive();
        b.arrive();
        assert_eq!(b.pending(), 2);
        b.reset();
        assert_eq!(b.pending(), 4);
    }

    #[test]
    fn test_barrier_is_complete_after_full_cycle() {
        let mut b = GridSyncBarrier::new(1).unwrap();
        assert!(!b.is_complete());
        b.arrive();
        assert!(b.is_complete());
    }

    #[test]
    fn test_barrier_single_block() {
        let mut b = GridSyncBarrier::new(1).unwrap();
        assert!(b.arrive());
        assert_eq!(b.generation, 1);
    }

    #[test]
    fn test_barrier_display_id() {
        let b = GridSyncBarrier::new(1).unwrap();
        let s = format!("{}", b.id);
        assert!(s.starts_with("barrier-"));
    }

    // ===== OccupancyConfig =====

    #[test]
    fn test_occupancy_config_default() {
        let cfg = OccupancyConfig::default();
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.dynamic_shared_mem, 0);
        assert_eq!(cfg.registers_per_thread, 0);
    }

    #[test]
    fn test_occupancy_config_new() {
        let cfg = OccupancyConfig::new(128).unwrap();
        assert_eq!(cfg.block_size, 128);
    }

    #[test]
    fn test_occupancy_config_zero() {
        assert!(OccupancyConfig::new(0).is_err());
    }

    #[test]
    fn test_occupancy_config_too_large() {
        assert!(OccupancyConfig::new(2048).is_err());
    }

    #[test]
    fn test_occupancy_config_with_shared_mem() {
        let cfg = OccupancyConfig::new(256).unwrap().with_shared_mem(4096);
        assert_eq!(cfg.dynamic_shared_mem, 4096);
    }

    #[test]
    fn test_occupancy_config_with_registers() {
        let cfg = OccupancyConfig::new(256).unwrap().with_registers(48);
        assert_eq!(cfg.registers_per_thread, 48);
    }

    // ===== max_active_blocks_per_sm =====

    #[test]
    fn test_occupancy_basic() {
        let caps = default_caps();
        let cfg = OccupancyConfig::new(256).unwrap();
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert!(result.max_active_blocks_per_sm > 0);
        assert!(result.occupancy_percent > 0);
        assert!(result.occupancy_percent <= 100);
        assert!(result.recommended_grid_size > 0);
    }

    #[test]
    fn test_occupancy_large_block() {
        let caps = default_caps();
        let cfg = OccupancyConfig::new(1024).unwrap();
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert!(result.max_active_blocks_per_sm >= 1);
    }

    #[test]
    fn test_occupancy_small_block() {
        let caps = default_caps();
        let cfg = OccupancyConfig::new(32).unwrap();
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert!(result.max_active_blocks_per_sm > 1);
    }

    #[test]
    fn test_occupancy_with_shared_mem() {
        let caps = default_caps();
        let cfg = OccupancyConfig::new(256).unwrap().with_shared_mem(32768);
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert!(result.max_active_blocks_per_sm >= 1);
    }

    #[test]
    fn test_occupancy_with_registers() {
        let caps = default_caps();
        let cfg = OccupancyConfig::new(256).unwrap().with_registers(64);
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert!(result.max_active_blocks_per_sm >= 1);
    }

    #[test]
    fn test_occupancy_zero_block_size_error() {
        let caps = default_caps();
        let cfg = OccupancyConfig { block_size: 0, ..Default::default() };
        assert!(max_active_blocks_per_sm(&cfg, &caps).is_err());
    }

    #[test]
    fn test_occupancy_exceeds_device_max() {
        let mut caps = default_caps();
        caps.max_threads_per_block = 128;
        let cfg = OccupancyConfig { block_size: 256, ..Default::default() };
        assert!(max_active_blocks_per_sm(&cfg, &caps).is_err());
    }

    #[test]
    fn test_occupancy_recommended_grid_uses_sm_count() {
        let caps = default_caps();
        let cfg = OccupancyConfig::new(256).unwrap();
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert_eq!(result.recommended_grid_size, result.max_active_blocks_per_sm * caps.sm_count);
    }

    // ===== compute_cooperative_launch_config =====

    #[test]
    fn test_compute_config_basic() {
        let caps = default_caps();
        let occ = OccupancyConfig::new(256).unwrap();
        let cfg = compute_cooperative_launch_config(1024, &occ, &caps).unwrap();
        assert!(cfg.grid[0] > 0);
        assert_eq!(cfg.block[0], 256);
        assert!(cfg.cooperative);
    }

    #[test]
    fn test_compute_config_large_data() {
        let caps = default_caps();
        let occ = OccupancyConfig::new(256).unwrap();
        let cfg = compute_cooperative_launch_config(10_000_000, &occ, &caps).unwrap();
        // Grid should be clamped to max cooperative blocks
        let max_coop =
            max_active_blocks_per_sm(&occ, &caps).unwrap().max_active_blocks_per_sm * caps.sm_count;
        assert!(cfg.grid[0] <= max_coop);
    }

    #[test]
    fn test_compute_config_small_data() {
        let caps = default_caps();
        let occ = OccupancyConfig::new(256).unwrap();
        let cfg = compute_cooperative_launch_config(100, &occ, &caps).unwrap();
        assert_eq!(cfg.grid[0], 1);
    }

    #[test]
    fn test_compute_config_zero_data() {
        let caps = default_caps();
        let occ = OccupancyConfig::new(256).unwrap();
        assert!(compute_cooperative_launch_config(0, &occ, &caps).is_err());
    }

    #[test]
    fn test_compute_config_exact_block_boundary() {
        let caps = default_caps();
        let occ = OccupancyConfig::new(256).unwrap();
        let cfg = compute_cooperative_launch_config(256, &occ, &caps).unwrap();
        assert_eq!(cfg.grid[0], 1);
    }

    #[test]
    fn test_compute_config_one_over_boundary() {
        let caps = default_caps();
        let occ = OccupancyConfig::new(256).unwrap();
        let cfg = compute_cooperative_launch_config(257, &occ, &caps).unwrap();
        assert_eq!(cfg.grid[0], 2);
    }

    // ===== BlockClusterConfig =====

    #[test]
    fn test_cluster_new_valid() {
        let c = BlockClusterConfig::new(4).unwrap();
        assert_eq!(c.cluster_size, 4);
        assert!(!c.distributed_shared_mem);
    }

    #[test]
    fn test_cluster_new_size_1() {
        let c = BlockClusterConfig::new(1).unwrap();
        assert_eq!(c.cluster_size, 1);
    }

    #[test]
    fn test_cluster_new_size_16() {
        let c = BlockClusterConfig::new(16).unwrap();
        assert_eq!(c.cluster_size, 16);
    }

    #[test]
    fn test_cluster_new_zero() {
        assert!(BlockClusterConfig::new(0).is_err());
    }

    #[test]
    fn test_cluster_new_too_large() {
        assert!(BlockClusterConfig::new(32).is_err());
    }

    #[test]
    fn test_cluster_new_not_power_of_two() {
        assert!(BlockClusterConfig::new(3).is_err());
        assert!(BlockClusterConfig::new(6).is_err());
    }

    #[test]
    fn test_cluster_with_distributed_shared_mem() {
        let c = BlockClusterConfig::new(4).unwrap().with_distributed_shared_mem(65536);
        assert!(c.distributed_shared_mem);
        assert_eq!(c.total_cluster_shared_mem, 65536);
    }

    #[test]
    fn test_cluster_with_grid() {
        let c = BlockClusterConfig::new(2).unwrap().with_cluster_grid([4, 2, 1]).unwrap();
        assert_eq!(c.cluster_grid, [4, 2, 1]);
    }

    #[test]
    fn test_cluster_with_grid_zero() {
        assert!(BlockClusterConfig::new(2).unwrap().with_cluster_grid([0, 1, 1]).is_err());
    }

    #[test]
    fn test_cluster_total_blocks() {
        let c = BlockClusterConfig::new(4).unwrap().with_cluster_grid([8, 1, 1]).unwrap();
        assert_eq!(c.total_cluster_blocks(), 32); // 8 * 1 * 1 * 4
    }

    #[test]
    fn test_cluster_validate_hopper() {
        let caps = hopper_caps();
        let c = BlockClusterConfig::new(4).unwrap();
        assert!(c.validate(&caps).is_ok());
    }

    #[test]
    fn test_cluster_validate_ampere_fails() {
        let caps = default_caps(); // Ampere doesn't support clusters
        let c = BlockClusterConfig::new(4).unwrap();
        assert!(c.validate(&caps).is_err());
    }

    #[test]
    fn test_cluster_default() {
        let c = BlockClusterConfig::default();
        assert_eq!(c.cluster_size, 2);
        assert!(!c.distributed_shared_mem);
    }

    // ===== compute_cluster_launch_config =====

    #[test]
    fn test_cluster_launch_basic() {
        let caps = hopper_caps();
        let cluster = BlockClusterConfig::new(4).unwrap();
        let cfg = compute_cluster_launch_config(4096, 256, &cluster, &caps).unwrap();
        assert!(cfg.grid[0] > 0);
        assert!(cfg.grid[0] % cluster.cluster_size == 0);
    }

    #[test]
    fn test_cluster_launch_zero_data() {
        let caps = hopper_caps();
        let cluster = BlockClusterConfig::new(4).unwrap();
        assert!(compute_cluster_launch_config(0, 256, &cluster, &caps).is_err());
    }

    #[test]
    fn test_cluster_launch_zero_block_size() {
        let caps = hopper_caps();
        let cluster = BlockClusterConfig::new(4).unwrap();
        assert!(compute_cluster_launch_config(1024, 0, &cluster, &caps).is_err());
    }

    #[test]
    fn test_cluster_launch_unsupported_device() {
        let caps = default_caps(); // Ampere
        let cluster = BlockClusterConfig::new(4).unwrap();
        assert!(compute_cluster_launch_config(1024, 256, &cluster, &caps).is_err());
    }

    // ===== MultiGridLaunchConfig =====

    #[test]
    fn test_multi_grid_single_device() {
        let caps = vec![default_caps()];
        let cfg =
            MultiGridLaunchConfig::new(1024, 256, &caps, MultiGridSyncMode::FullSync).unwrap();
        assert_eq!(cfg.num_devices(), 1);
        assert_eq!(cfg.device_configs[0].data_range, (0, 1024));
    }

    #[test]
    fn test_multi_grid_two_devices() {
        let caps = vec![default_caps(), default_caps()];
        let cfg =
            MultiGridLaunchConfig::new(1000, 256, &caps, MultiGridSyncMode::FullSync).unwrap();
        assert_eq!(cfg.num_devices(), 2);
        let (s0, c0) = cfg.device_configs[0].data_range;
        let (s1, c1) = cfg.device_configs[1].data_range;
        assert_eq!(s0, 0);
        assert_eq!(c0, 500);
        assert_eq!(s1, 500);
        assert_eq!(c1, 500);
    }

    #[test]
    fn test_multi_grid_four_devices() {
        let caps = vec![default_caps(); 4];
        let cfg =
            MultiGridLaunchConfig::new(4096, 256, &caps, MultiGridSyncMode::Independent).unwrap();
        assert_eq!(cfg.num_devices(), 4);
        assert_eq!(
            cfg.total_threads(),
            cfg.device_configs.iter().map(|d| d.config.total_threads()).sum::<u64>()
        );
    }

    #[test]
    fn test_multi_grid_no_devices() {
        let caps: Vec<DeviceCooperativeCapabilities> = vec![];
        assert!(MultiGridLaunchConfig::new(1024, 256, &caps, MultiGridSyncMode::FullSync).is_err());
    }

    #[test]
    fn test_multi_grid_zero_data() {
        let caps = vec![default_caps()];
        assert!(MultiGridLaunchConfig::new(0, 256, &caps, MultiGridSyncMode::FullSync).is_err());
    }

    #[test]
    fn test_multi_grid_zero_block_size() {
        let caps = vec![default_caps()];
        assert!(MultiGridLaunchConfig::new(1024, 0, &caps, MultiGridSyncMode::FullSync).is_err());
    }

    #[test]
    fn test_multi_grid_pipeline_sync() {
        let caps = vec![default_caps(); 2];
        let cfg =
            MultiGridLaunchConfig::new(2048, 256, &caps, MultiGridSyncMode::PipelineSync).unwrap();
        assert_eq!(cfg.sync_mode, MultiGridSyncMode::PipelineSync);
    }

    #[test]
    fn test_multi_grid_total_threads() {
        let caps = vec![default_caps()];
        let cfg =
            MultiGridLaunchConfig::new(1024, 256, &caps, MultiGridSyncMode::FullSync).unwrap();
        assert!(cfg.total_threads() > 0);
    }

    // ===== grid_stride_loop =====

    #[test]
    fn test_grid_stride_loop_scale() {
        let config = CooperativeLaunchConfig::new_1d(2, 4).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        grid_stride_loop(&mut data, &config, |v, _| v * 2.0).unwrap();
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_grid_stride_loop_identity() {
        let config = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        let mut data = vec![42.0; 100];
        grid_stride_loop(&mut data, &config, |v, _| v).unwrap();
        assert!(data.iter().all(|&v| (v - 42.0).abs() < 1e-6));
    }

    #[test]
    fn test_grid_stride_loop_with_index() {
        let config = CooperativeLaunchConfig::new_1d(1, 4).unwrap();
        let mut data = vec![0.0; 8];
        grid_stride_loop(&mut data, &config, |_, i| i as f32).unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_grid_stride_loop_large_data_small_grid() {
        let config = CooperativeLaunchConfig::new_1d(1, 2).unwrap();
        let mut data = vec![1.0; 100];
        grid_stride_loop(&mut data, &config, |v, _| v + 1.0).unwrap();
        assert!(data.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_grid_stride_loop_empty() {
        let config = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        let mut data: Vec<f32> = vec![];
        assert!(grid_stride_loop(&mut data, &config, |v, _| v).is_err());
    }

    #[test]
    fn test_grid_stride_loop_single_element() {
        let config = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        let mut data = vec![5.0];
        grid_stride_loop(&mut data, &config, |v, _| v * 3.0).unwrap();
        assert!((data[0] - 15.0).abs() < 1e-6);
    }

    // ===== grid_stride_loop_2d =====

    #[test]
    fn test_grid_stride_2d_basic() {
        let config = CooperativeLaunchConfig::new_2d([2, 2], [2, 2]).unwrap();
        let mut data = vec![1.0; 16];
        grid_stride_loop_2d(&mut data, 4, 4, &config, |v, _r, _c| v + 1.0).unwrap();
        assert!(data.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_grid_stride_2d_with_coords() {
        let config = CooperativeLaunchConfig::new_2d([1, 1], [4, 4]).unwrap();
        let mut data = vec![0.0; 16];
        grid_stride_loop_2d(&mut data, 4, 4, &config, |_, r, c| (r * 4 + c) as f32).unwrap();
        for i in 0..16 {
            assert!((data[i] - i as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_grid_stride_2d_empty() {
        let config = CooperativeLaunchConfig::new_2d([1, 1], [1, 1]).unwrap();
        let mut data: Vec<f32> = vec![];
        assert!(grid_stride_loop_2d(&mut data, 0, 0, &config, |v, _, _| v).is_err());
    }

    #[test]
    fn test_grid_stride_2d_dimension_mismatch() {
        let config = CooperativeLaunchConfig::new_2d([1, 1], [1, 1]).unwrap();
        let mut data = vec![1.0; 10];
        assert!(grid_stride_loop_2d(&mut data, 3, 4, &config, |v, _, _| v).is_err());
    }

    // ===== launch_cooperative_kernel =====

    #[test]
    fn test_launch_cooperative_scale() {
        let config = CooperativeLaunchConfig::new_1d(4, 256).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let result = launch_cooperative_kernel(&mut data, &config, |d, _barrier| {
            for v in d.iter_mut() {
                *v *= 10.0;
            }
        })
        .unwrap();
        assert_eq!(data, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(result.elements_processed, 4);
        assert!(result.grid_sync_used);
        assert_eq!(result.blocks_executed, 4);
    }

    #[test]
    fn test_launch_cooperative_with_barrier() {
        let config = CooperativeLaunchConfig::new_1d(2, 256).unwrap();
        let mut data = vec![1.0; 8];
        let result = launch_cooperative_kernel(&mut data, &config, |d, barrier| {
            // Phase 1: scale
            for v in d.iter_mut() {
                *v *= 2.0;
            }
            // Simulate barrier (all blocks arrive)
            for _ in 0..2 {
                barrier.arrive();
            }
            // Phase 2: add
            for v in d.iter_mut() {
                *v += 1.0;
            }
        })
        .unwrap();
        assert!(data.iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert_eq!(result.blocks_executed, 2);
    }

    #[test]
    fn test_launch_cooperative_empty() {
        let config = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        let mut data: Vec<f32> = vec![];
        assert!(launch_cooperative_kernel(&mut data, &config, |_, _| {}).is_err());
    }

    #[test]
    fn test_launch_id_increments() {
        let config = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        let mut d1 = vec![1.0];
        let mut d2 = vec![1.0];
        let r1 = launch_cooperative_kernel(&mut d1, &config, |_, _| {}).unwrap();
        let r2 = launch_cooperative_kernel(&mut d2, &config, |_, _| {}).unwrap();
        assert_ne!(r1.id, r2.id);
    }

    #[test]
    fn test_launch_id_display() {
        let config = CooperativeLaunchConfig::new_1d(1, 256).unwrap();
        let mut data = vec![1.0];
        let result = launch_cooperative_kernel(&mut data, &config, |_, _| {}).unwrap();
        let s = format!("{}", result.id);
        assert!(s.starts_with("coop-launch-"));
    }

    #[test]
    fn test_launch_non_cooperative() {
        let config = CooperativeLaunchConfig::new_1d(1, 256).unwrap().non_cooperative();
        let mut data = vec![1.0, 2.0];
        let result = launch_cooperative_kernel(&mut data, &config, |d, _| {
            for v in d.iter_mut() {
                *v += 1.0;
            }
        })
        .unwrap();
        assert!(!result.grid_sync_used);
    }

    // ===== launch_multi_grid_cooperative =====

    #[test]
    fn test_multi_grid_launch_basic() {
        let caps = vec![default_caps(); 2];
        let multi = MultiGridLaunchConfig::new(8, 4, &caps, MultiGridSyncMode::FullSync).unwrap();
        let mut data = vec![1.0; 8];
        let results = launch_multi_grid_cooperative(&mut data, &multi, |slice, device_id| {
            for v in slice.iter_mut() {
                *v += device_id as f32;
            }
        })
        .unwrap();
        assert_eq!(results.len(), 2);
        // Device 0 processes first half, device 1 the second
        assert!(data[0..4].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(data[4..8].iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_multi_grid_launch_empty() {
        let caps = vec![default_caps()];
        let multi = MultiGridLaunchConfig::new(4, 4, &caps, MultiGridSyncMode::FullSync).unwrap();
        let mut data: Vec<f32> = vec![];
        assert!(launch_multi_grid_cooperative(&mut data, &multi, |_, _| {}).is_err());
    }

    #[test]
    fn test_multi_grid_launch_out_of_bounds() {
        // Manually create a config with bad range
        let multi = MultiGridLaunchConfig {
            device_configs: vec![DeviceLaunchEntry {
                device_id: 0,
                config: CooperativeLaunchConfig::new_1d(1, 4).unwrap(),
                data_range: (0, 100),
            }],
            sync_mode: MultiGridSyncMode::FullSync,
        };
        let mut data = vec![1.0; 10];
        assert!(launch_multi_grid_cooperative(&mut data, &multi, |_, _| {}).is_err());
    }

    // ===== CUDA kernel source =====

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn test_kernel_src_not_empty() {
        assert!(!COOPERATIVE_LAUNCH_KERNEL_SRC.is_empty());
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn test_kernel_src_contains_grid_sync() {
        assert!(COOPERATIVE_LAUNCH_KERNEL_SRC.contains("grid.sync()"));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn test_kernel_src_contains_cooperative_groups() {
        assert!(COOPERATIVE_LAUNCH_KERNEL_SRC.contains("cooperative_groups"));
    }

    // ===== Edge cases and stress tests =====

    #[test]
    fn test_config_max_block_size_1024() {
        let cfg = CooperativeLaunchConfig::new_1d(1, 1024).unwrap();
        assert_eq!(cfg.threads_per_block(), 1024);
    }

    #[test]
    fn test_config_min_grid_and_block() {
        let cfg = CooperativeLaunchConfig::new_1d(1, 1).unwrap();
        assert_eq!(cfg.total_threads(), 1);
    }

    #[test]
    fn test_occupancy_hopper() {
        let caps = hopper_caps();
        let cfg = OccupancyConfig::new(256).unwrap();
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert!(result.max_active_blocks_per_sm > 0);
        assert!(result.recommended_grid_size > 0);
    }

    #[test]
    fn test_grid_stride_loop_non_divisible() {
        let config = CooperativeLaunchConfig::new_1d(1, 3).unwrap();
        let mut data = vec![1.0; 10];
        grid_stride_loop(&mut data, &config, |v, _| v + 1.0).unwrap();
        assert!(data.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_barrier_id_uniqueness() {
        let b1 = GridSyncBarrier::new(1).unwrap();
        let b2 = GridSyncBarrier::new(1).unwrap();
        assert_ne!(b1.id, b2.id);
    }

    #[test]
    fn test_sync_mode_equality() {
        assert_eq!(MultiGridSyncMode::FullSync, MultiGridSyncMode::FullSync);
        assert_ne!(MultiGridSyncMode::FullSync, MultiGridSyncMode::PipelineSync);
        assert_ne!(MultiGridSyncMode::FullSync, MultiGridSyncMode::Independent);
    }

    #[test]
    fn test_device_launch_entry_clone() {
        let entry = DeviceLaunchEntry {
            device_id: 0,
            config: CooperativeLaunchConfig::default(),
            data_range: (0, 100),
        };
        let entry2 = entry.clone();
        assert_eq!(entry.device_id, entry2.device_id);
        assert_eq!(entry.data_range, entry2.data_range);
    }

    #[test]
    fn test_multi_grid_uneven_distribution() {
        let caps = vec![default_caps(); 3];
        let cfg = MultiGridLaunchConfig::new(10, 4, &caps, MultiGridSyncMode::Independent).unwrap();
        let total: usize = cfg.device_configs.iter().map(|e| e.data_range.1).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_cluster_launch_single_block_per_cluster() {
        let caps = hopper_caps();
        let cluster = BlockClusterConfig::new(1).unwrap();
        let cfg = compute_cluster_launch_config(256, 256, &cluster, &caps).unwrap();
        assert_eq!(cfg.grid[0], 1);
    }

    #[test]
    fn test_compute_config_with_shared_mem() {
        let caps = default_caps();
        let occ = OccupancyConfig::new(256).unwrap().with_shared_mem(8192);
        let cfg = compute_cooperative_launch_config(1024, &occ, &caps).unwrap();
        assert_eq!(cfg.shared_mem_bytes, 8192);
    }

    #[test]
    fn test_multi_grid_large_block_size_error() {
        let caps = vec![default_caps()];
        assert!(
            MultiGridLaunchConfig::new(1024, 2048, &caps, MultiGridSyncMode::FullSync).is_err()
        );
    }

    #[test]
    fn test_grid_stride_loop_exact_stride() {
        let config = CooperativeLaunchConfig::new_1d(2, 2).unwrap();
        let mut data = vec![0.0; 4];
        grid_stride_loop(&mut data, &config, |_, i| i as f32).unwrap();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cluster_block_size_exceeds_limit() {
        let caps = hopper_caps();
        let cluster = BlockClusterConfig::new(4).unwrap();
        assert!(compute_cluster_launch_config(1024, 2048, &cluster, &caps).is_err());
    }

    #[test]
    fn test_occupancy_result_fields() {
        let caps = default_caps();
        let cfg = OccupancyConfig::new(256).unwrap();
        let result = max_active_blocks_per_sm(&cfg, &caps).unwrap();
        assert!(result.active_warps_per_sm > 0);
        assert!(result.max_warps_per_sm > 0);
        assert!(result.active_warps_per_sm <= result.max_warps_per_sm);
    }
}
