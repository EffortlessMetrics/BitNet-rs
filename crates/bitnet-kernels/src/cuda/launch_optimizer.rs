//! CUDA kernel launch optimization with occupancy-based tuning and caching.
//!
//! # Overview
//!
//! This module provides a comprehensive kernel launch parameter optimizer that
//! eliminates redundant GPU configuration calculations and reduces CPU-GPU
//! synchronization overhead. Key components:
//!
//! - [`GpuArchitecture`] — GPU architecture descriptor with SM limits
//! - [`LaunchConfig`] — computed grid/block/shared-memory launch parameters
//! - [`LaunchOptimizer`] — occupancy-based block size selection with caching
//! - [`ReductionLaunchPlanner`] — progressive grid shrinking for reduction kernels
//! - [`PersistentKernelConfig`] — grid-stride loop launch patterns
//! - [`KernelBatch`] / [`BatchBuilder`] — multi-kernel batching to reduce sync
//! - [`SharedMemoryPlanner`] — dynamic shared memory sizing
//! - [`LaunchValidator`] — launch parameter validation and error reporting
//!
//! All GPU dispatch is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations allow testing on non-GPU hosts.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use bitnet_common::{KernelError, Result};

// ── GpuArchitecture ──────────────────────────────────────────────────

/// Known GPU architecture families with their SM resource limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchFamily {
    /// SM 6.x — Pascal (GTX 10xx, Tesla P100).
    Pascal,
    /// SM 7.0 — Volta (Tesla V100).
    Volta,
    /// SM 7.5 — Turing (RTX 20xx, Tesla T4).
    Turing,
    /// SM 8.0/8.6 — Ampere (RTX 30xx, A100).
    Ampere,
    /// SM 8.9 — Ada Lovelace (RTX 40xx, L40).
    Ada,
    /// SM 9.0 — Hopper (H100).
    Hopper,
}

impl fmt::Display for GpuArchFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pascal => write!(f, "Pascal (SM 6.x)"),
            Self::Volta => write!(f, "Volta (SM 7.0)"),
            Self::Turing => write!(f, "Turing (SM 7.5)"),
            Self::Ampere => write!(f, "Ampere (SM 8.x)"),
            Self::Ada => write!(f, "Ada Lovelace (SM 8.9)"),
            Self::Hopper => write!(f, "Hopper (SM 9.0)"),
        }
    }
}

/// GPU architecture descriptor with per-SM resource limits.
#[derive(Debug, Clone)]
pub struct GpuArchitecture {
    /// Architecture family.
    pub family: GpuArchFamily,
    /// Compute capability major.minor (e.g. 8.0 for Ampere).
    pub compute_capability: (u32, u32),
    /// Number of SMs on the device.
    pub sm_count: u32,
    /// Maximum threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum blocks (thread-block clusters) per SM.
    pub max_blocks_per_sm: u32,
    /// Maximum shared memory per SM in bytes.
    pub max_shared_mem_per_sm: u32,
    /// Maximum shared memory per block in bytes.
    pub max_shared_mem_per_block: u32,
    /// Maximum registers per SM.
    pub max_registers_per_sm: u32,
    /// Maximum registers per block.
    pub max_registers_per_block: u32,
    /// Warp size (always 32 for NVIDIA).
    pub warp_size: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Maximum block dimensions [x, y, z].
    pub max_block_dim: [u32; 3],
    /// Maximum grid dimensions [x, y, z].
    pub max_grid_dim: [u32; 3],
}

impl GpuArchitecture {
    /// Create architecture descriptor from compute capability.
    pub fn from_compute_capability(major: u32, minor: u32, sm_count: u32) -> Result<Self> {
        let (
            family,
            max_threads_per_sm,
            max_blocks_per_sm,
            max_shared_sm,
            max_shared_block,
            max_regs_sm,
        ) = match (major, minor) {
            (6, _) => (GpuArchFamily::Pascal, 2048, 32, 65_536, 49_152, 65_536),
            (7, 0) => (GpuArchFamily::Volta, 2048, 32, 98_304, 49_152, 65_536),
            (7, 5) => (GpuArchFamily::Turing, 1024, 16, 65_536, 49_152, 65_536),
            (8, 0) => (GpuArchFamily::Ampere, 2048, 32, 163_840, 49_152, 65_536),
            (8, 6) => (GpuArchFamily::Ampere, 1536, 16, 102_400, 49_152, 65_536),
            (8, 9) => (GpuArchFamily::Ada, 1536, 24, 102_400, 49_152, 65_536),
            (9, 0) => (GpuArchFamily::Hopper, 2048, 32, 233_472, 49_152, 65_536),
            _ => {
                return Err(KernelError::UnsupportedArchitecture {
                    arch: format!("SM {major}.{minor}"),
                }
                .into());
            }
        };

        Ok(Self {
            family,
            compute_capability: (major, minor),
            sm_count,
            max_threads_per_sm,
            max_blocks_per_sm,
            max_shared_mem_per_sm: max_shared_sm,
            max_shared_mem_per_block: max_shared_block,
            max_registers_per_sm: max_regs_sm,
            max_registers_per_block: max_regs_sm,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [2_147_483_647, 65_535, 65_535],
        })
    }

    /// Default Ampere SM 8.0 (A100) with 108 SMs.
    pub fn ampere_a100() -> Self {
        Self::from_compute_capability(8, 0, 108).expect("known arch")
    }

    /// Default Hopper SM 9.0 (H100) with 132 SMs.
    pub fn hopper_h100() -> Self {
        Self::from_compute_capability(9, 0, 132).expect("known arch")
    }

    /// Maximum warps per SM.
    pub fn max_warps_per_sm(&self) -> u32 {
        self.max_threads_per_sm / self.warp_size
    }

    /// Whether the architecture supports tensor cores.
    pub fn has_tensor_cores(&self) -> bool {
        self.compute_capability.0 >= 7
    }

    /// Whether the architecture supports FP16 natively.
    pub fn has_native_fp16(&self) -> bool {
        self.compute_capability.0 >= 6
    }

    /// Whether the architecture supports async copy (`cp.async`).
    pub fn has_async_copy(&self) -> bool {
        self.compute_capability.0 >= 8
    }
}

impl Default for GpuArchitecture {
    fn default() -> Self {
        Self::ampere_a100()
    }
}

// ── LaunchConfig ─────────────────────────────────────────────────────

/// Computed kernel launch configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks).
    pub grid: [u32; 3],
    /// Block dimensions (threads).
    pub block: [u32; 3],
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Create a 1-D launch configuration.
    pub fn new_1d(grid_x: u32, block_x: u32) -> Self {
        Self { grid: [grid_x, 1, 1], block: [block_x, 1, 1], shared_mem_bytes: 0 }
    }

    /// Create a 2-D launch configuration.
    pub fn new_2d(grid: [u32; 2], block: [u32; 2]) -> Self {
        Self { grid: [grid[0], grid[1], 1], block: [block[0], block[1], 1], shared_mem_bytes: 0 }
    }

    /// Create a 3-D launch configuration.
    pub fn new_3d(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self { grid, block, shared_mem_bytes: 0 }
    }

    /// Attach dynamic shared memory.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Total number of threads across the grid.
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
}

impl fmt::Display for LaunchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "<<<({},{},{}), ({},{},{})>>> smem={}B",
            self.grid[0],
            self.grid[1],
            self.grid[2],
            self.block[0],
            self.block[1],
            self.block[2],
            self.shared_mem_bytes
        )
    }
}

// ── OccupancyInfo ────────────────────────────────────────────────────

/// Occupancy details for a chosen block size.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyInfo {
    /// Theoretical occupancy (0.0–1.0).
    pub occupancy: f64,
    /// Active warps per SM.
    pub active_warps_per_sm: u32,
    /// Maximum warps per SM.
    pub max_warps_per_sm: u32,
    /// Active blocks per SM.
    pub active_blocks_per_sm: u32,
    /// Limiting resource.
    pub limiter: OccupancyLimiter,
}

/// What resource limits occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLimiter {
    /// Thread count is the bottleneck.
    Threads,
    /// Block count is the bottleneck.
    Blocks,
    /// Shared memory usage is the bottleneck.
    SharedMemory,
    /// Register usage is the bottleneck.
    Registers,
}

impl fmt::Display for OccupancyLimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Threads => write!(f, "threads"),
            Self::Blocks => write!(f, "blocks"),
            Self::SharedMemory => write!(f, "shared_memory"),
            Self::Registers => write!(f, "registers"),
        }
    }
}

// ── KernelResourceUsage ──────────────────────────────────────────────

/// Per-kernel resource requirements used for occupancy calculation.
#[derive(Debug, Clone)]
pub struct KernelResourceUsage {
    /// Registers per thread.
    pub registers_per_thread: u32,
    /// Static shared memory in bytes.
    pub static_shared_mem: u32,
    /// Dynamic shared memory per block (set by launch).
    pub dynamic_shared_mem: u32,
}

impl Default for KernelResourceUsage {
    fn default() -> Self {
        Self { registers_per_thread: 32, static_shared_mem: 0, dynamic_shared_mem: 0 }
    }
}

// ── CacheKey ─────────────────────────────────────────────────────────

/// Key for the launch config cache (kernel name + problem dimensions).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    kernel_name: String,
    problem_size: [u64; 3],
}

// ── LaunchOptimizer ──────────────────────────────────────────────────

/// Occupancy-based kernel launch optimizer with caching.
///
/// Computes optimal grid/block dimensions for a given GPU architecture and
/// kernel resource usage, then caches results to avoid recomputation on
/// repeated launches with the same problem shape.
#[derive(Debug)]
pub struct LaunchOptimizer {
    /// Target GPU architecture.
    arch: GpuArchitecture,
    /// Cached launch configurations.
    cache: HashMap<CacheKey, LaunchConfig>,
    /// Cache hit counter.
    cache_hits: u64,
    /// Cache miss counter.
    cache_misses: u64,
}

impl LaunchOptimizer {
    /// Create an optimizer for the given architecture.
    pub fn new(arch: GpuArchitecture) -> Self {
        Self { arch, cache: HashMap::new(), cache_hits: 0, cache_misses: 0 }
    }

    /// Access the target architecture.
    pub fn architecture(&self) -> &GpuArchitecture {
        &self.arch
    }

    /// Number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Cache hit count.
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    /// Cache miss count.
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses
    }

    /// Cache hit rate (0.0–1.0). Returns 0 if no lookups.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }

    /// Clear all cached configurations.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Compute optimal 1-D launch config for an element-wise kernel.
    ///
    /// Selects the block size that maximizes occupancy, then computes the
    /// grid size to cover `n` elements.
    pub fn optimize_1d(
        &mut self,
        kernel_name: &str,
        n: u64,
        resources: &KernelResourceUsage,
    ) -> Result<LaunchConfig> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "problem size must be > 0".into(),
            }
            .into());
        }

        let key = CacheKey { kernel_name: kernel_name.to_string(), problem_size: [n, 1, 1] };
        if let Some(cfg) = self.cache.get(&key) {
            self.cache_hits += 1;
            return Ok(cfg.clone());
        }
        self.cache_misses += 1;

        let block_size = self.optimal_block_size_1d(resources);
        let grid_x = n.div_ceil(block_size as u64) as u32;
        let grid_x = grid_x.min(self.arch.max_grid_dim[0]);

        let config =
            LaunchConfig::new_1d(grid_x, block_size).with_shared_mem(resources.dynamic_shared_mem);

        self.cache.insert(key, config.clone());
        Ok(config)
    }

    /// Compute optimal 2-D launch config (e.g., matrix operations).
    pub fn optimize_2d(
        &mut self,
        kernel_name: &str,
        rows: u64,
        cols: u64,
        resources: &KernelResourceUsage,
    ) -> Result<LaunchConfig> {
        if rows == 0 || cols == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "rows and cols must be > 0".into(),
            }
            .into());
        }

        let key = CacheKey { kernel_name: kernel_name.to_string(), problem_size: [rows, cols, 1] };
        if let Some(cfg) = self.cache.get(&key) {
            self.cache_hits += 1;
            return Ok(cfg.clone());
        }
        self.cache_misses += 1;

        let total_threads = self.optimal_block_size_1d(resources);
        let (bx, by) = Self::factor_block_2d(total_threads);

        let gx = cols.div_ceil(bx as u64) as u32;
        let gy = rows.div_ceil(by as u64) as u32;
        let gx = gx.min(self.arch.max_grid_dim[0]);
        let gy = gy.min(self.arch.max_grid_dim[1]);

        let config =
            LaunchConfig::new_2d([gx, gy], [bx, by]).with_shared_mem(resources.dynamic_shared_mem);

        self.cache.insert(key, config.clone());
        Ok(config)
    }

    /// Compute optimal 3-D launch config (e.g., batched operations).
    pub fn optimize_3d(
        &mut self,
        kernel_name: &str,
        dim: [u64; 3],
        resources: &KernelResourceUsage,
    ) -> Result<LaunchConfig> {
        if dim[0] == 0 || dim[1] == 0 || dim[2] == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "all dimensions must be > 0".into(),
            }
            .into());
        }

        let key = CacheKey { kernel_name: kernel_name.to_string(), problem_size: dim };
        if let Some(cfg) = self.cache.get(&key) {
            self.cache_hits += 1;
            return Ok(cfg.clone());
        }
        self.cache_misses += 1;

        let total_threads = self.optimal_block_size_1d(resources);
        let (bx, by, bz) = Self::factor_block_3d(total_threads);

        let gx = dim[0].div_ceil(bx as u64) as u32;
        let gy = dim[1].div_ceil(by as u64) as u32;
        let gz = dim[2].div_ceil(bz as u64) as u32;
        let gx = gx.min(self.arch.max_grid_dim[0]);
        let gy = gy.min(self.arch.max_grid_dim[1]);
        let gz = gz.min(self.arch.max_grid_dim[2]);

        let config = LaunchConfig::new_3d([gx, gy, gz], [bx, by, bz])
            .with_shared_mem(resources.dynamic_shared_mem);

        self.cache.insert(key, config.clone());
        Ok(config)
    }

    /// Select the block size (power of 2, 32..=1024) that maximizes occupancy.
    fn optimal_block_size_1d(&self, resources: &KernelResourceUsage) -> u32 {
        let candidates = [32, 64, 128, 256, 512, 1024];
        let mut best_block = 256u32;
        let mut best_occ = 0.0f64;

        for &bs in &candidates {
            if bs > self.arch.max_threads_per_block {
                break;
            }
            let info = self.compute_occupancy(bs, resources);
            if info.occupancy > best_occ {
                best_occ = info.occupancy;
                best_block = bs;
            }
        }
        best_block
    }

    /// Compute occupancy for a given block size and resource usage.
    pub fn compute_occupancy(
        &self,
        threads_per_block: u32,
        resources: &KernelResourceUsage,
    ) -> OccupancyInfo {
        let warp_size = self.arch.warp_size;
        let max_warps = self.arch.max_warps_per_sm();
        let warps_per_block = threads_per_block.div_ceil(warp_size);

        if warps_per_block == 0 || threads_per_block == 0 {
            return OccupancyInfo {
                occupancy: 0.0,
                active_warps_per_sm: 0,
                max_warps_per_sm: max_warps,
                active_blocks_per_sm: 0,
                limiter: OccupancyLimiter::Threads,
            };
        }

        let blocks_by_threads = self.arch.max_threads_per_sm / threads_per_block;
        let blocks_by_blocks = self.arch.max_blocks_per_sm;

        let total_shared = resources.static_shared_mem + resources.dynamic_shared_mem;
        let blocks_by_shared = if total_shared == 0 {
            self.arch.max_blocks_per_sm
        } else {
            self.arch.max_shared_mem_per_sm / total_shared
        };

        let regs_per_block = resources.registers_per_thread * threads_per_block;
        let blocks_by_regs = if regs_per_block == 0 {
            self.arch.max_blocks_per_sm
        } else {
            self.arch.max_registers_per_sm / regs_per_block
        };

        let limits = [
            (blocks_by_threads, OccupancyLimiter::Threads),
            (blocks_by_blocks, OccupancyLimiter::Blocks),
            (blocks_by_shared, OccupancyLimiter::SharedMemory),
            (blocks_by_regs, OccupancyLimiter::Registers),
        ];
        let (active_blocks, limiter) = limits
            .iter()
            .min_by_key(|(b, _)| *b)
            .copied()
            .unwrap_or((0, OccupancyLimiter::Threads));

        let active_warps = active_blocks * warps_per_block;
        let occupancy =
            if max_warps > 0 { (active_warps as f64 / max_warps as f64).min(1.0) } else { 0.0 };

        OccupancyInfo {
            occupancy,
            active_warps_per_sm: active_warps,
            max_warps_per_sm: max_warps,
            active_blocks_per_sm: active_blocks,
            limiter,
        }
    }

    /// Factor a total thread count into 2-D block dimensions.
    /// Prefers roughly square blocks (good for 2-D locality).
    fn factor_block_2d(total: u32) -> (u32, u32) {
        // Try largest square factor that divides evenly.
        let sqrt = (total as f64).sqrt() as u32;
        for by in (1..=sqrt).rev() {
            if total.is_multiple_of(by) {
                let bx = total / by;
                if bx <= 1024 && by <= 1024 {
                    return (bx, by);
                }
            }
        }
        (total, 1)
    }

    /// Factor a total thread count into 3-D block dimensions.
    fn factor_block_3d(total: u32) -> (u32, u32, u32) {
        // Allocate 4 threads to Z, distribute remainder in 2-D.
        let bz = if total >= 128 { 4 } else { 1 };
        let remaining = total / bz;
        let (bx, by) = Self::factor_block_2d(remaining);
        (bx, by, bz)
    }
}

// ── SharedMemoryPlanner ──────────────────────────────────────────────

/// Strategies for sizing dynamic shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedMemStrategy {
    /// No shared memory needed.
    None,
    /// One element per thread (e.g., reduction scratch).
    PerThread,
    /// Fixed number of bytes.
    Fixed(u32),
    /// Per-warp scratch (one element per warp).
    PerWarp,
    /// Tile-based: `tile_rows * tile_cols * elem_size`.
    Tiled { tile_rows: u32, tile_cols: u32 },
}

/// Plans dynamic shared memory allocation for kernel launches.
#[derive(Debug, Clone)]
pub struct SharedMemoryPlanner {
    /// Target architecture.
    arch: GpuArchitecture,
    /// Element size in bytes (default 4 for f32).
    pub elem_size: u32,
}

impl SharedMemoryPlanner {
    /// Create a planner for the given architecture.
    pub fn new(arch: GpuArchitecture) -> Self {
        Self { arch, elem_size: 4 }
    }

    /// Set element size (e.g., 2 for f16, 4 for f32).
    pub fn with_elem_size(mut self, size: u32) -> Self {
        self.elem_size = size;
        self
    }

    /// Compute required shared memory bytes for a strategy and block size.
    pub fn compute(&self, strategy: SharedMemStrategy, block_size: u32) -> Result<u32> {
        let bytes = match strategy {
            SharedMemStrategy::None => 0,
            SharedMemStrategy::PerThread => block_size * self.elem_size,
            SharedMemStrategy::Fixed(n) => n,
            SharedMemStrategy::PerWarp => {
                let warps = block_size.div_ceil(self.arch.warp_size);
                warps * self.elem_size
            }
            SharedMemStrategy::Tiled { tile_rows, tile_cols } => {
                tile_rows * tile_cols * self.elem_size
            }
        };

        if bytes > self.arch.max_shared_mem_per_block {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "shared memory {bytes}B exceeds per-block limit {}B",
                    self.arch.max_shared_mem_per_block
                ),
            }
            .into());
        }
        Ok(bytes)
    }

    /// Maximum tiles that fit in per-block shared memory.
    pub fn max_tile_size(&self) -> u32 {
        let total_elements = self.arch.max_shared_mem_per_block / self.elem_size;
        (total_elements as f64).sqrt() as u32
    }
}

// ── ReductionLaunchPlanner ───────────────────────────────────────────

/// Plans multi-pass reduction launches with progressive grid shrinking.
///
/// For a reduction of N elements, the first pass reduces to `grid_size` partial
/// results, the second pass reduces those, and so on until one element remains.
#[derive(Debug, Clone)]
pub struct ReductionLaunchPlanner {
    /// Block size for reduction kernels.
    pub block_size: u32,
    /// Elements processed per thread in each pass.
    pub elements_per_thread: u32,
}

/// A single pass in a multi-pass reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionPass {
    /// Grid size (number of blocks) for this pass.
    pub grid_size: u32,
    /// Block size (threads per block).
    pub block_size: u32,
    /// Input elements for this pass.
    pub input_size: u64,
    /// Output elements from this pass.
    pub output_size: u64,
    /// Shared memory bytes needed.
    pub shared_mem_bytes: u32,
}

impl Default for ReductionLaunchPlanner {
    fn default() -> Self {
        Self { block_size: 256, elements_per_thread: 4 }
    }
}

impl ReductionLaunchPlanner {
    /// Create with custom block size and elements-per-thread.
    pub fn new(block_size: u32, elements_per_thread: u32) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "block_size must be a positive power of 2".into(),
            }
            .into());
        }
        if elements_per_thread == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "elements_per_thread must be > 0".into(),
            }
            .into());
        }
        Ok(Self { block_size, elements_per_thread })
    }

    /// Plan all passes needed to reduce `n` elements to 1.
    pub fn plan(&self, n: u64) -> Result<Vec<ReductionPass>> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "cannot reduce 0 elements".into(),
            }
            .into());
        }
        if n == 1 {
            return Ok(vec![]);
        }

        let mut passes = Vec::new();
        let mut remaining = n;

        loop {
            let work_per_block = self.block_size as u64 * self.elements_per_thread as u64;
            let grid_size = remaining.div_ceil(work_per_block).max(1) as u32;
            let output_size = grid_size as u64;
            let shared_bytes = self.block_size * 4; // f32 scratch

            passes.push(ReductionPass {
                grid_size,
                block_size: self.block_size,
                input_size: remaining,
                output_size,
                shared_mem_bytes: shared_bytes,
            });

            if output_size <= 1 {
                break;
            }
            remaining = output_size;
        }
        Ok(passes)
    }

    /// Total number of passes needed for `n` elements.
    pub fn pass_count(&self, n: u64) -> usize {
        self.plan(n).map(|p| p.len()).unwrap_or(0)
    }
}

// ── PersistentKernelConfig ───────────────────────────────────────────

/// Configuration for persistent (grid-stride loop) kernel launches.
///
/// A persistent kernel launches exactly `sm_count * blocks_per_sm` blocks.
/// Each block processes multiple work items via a grid-stride loop, amortizing
/// launch overhead for many small work items.
#[derive(Debug, Clone)]
pub struct PersistentKernelConfig {
    /// Number of SMs on the device.
    pub sm_count: u32,
    /// Blocks per SM (typically 1–4 for persistent kernels).
    pub blocks_per_sm: u32,
    /// Threads per block.
    pub threads_per_block: u32,
    /// Total work items to process.
    pub total_work_items: u64,
}

impl PersistentKernelConfig {
    /// Create a persistent kernel config.
    pub fn new(
        sm_count: u32,
        blocks_per_sm: u32,
        threads_per_block: u32,
        total_work_items: u64,
    ) -> Result<Self> {
        if sm_count == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "sm_count must be > 0".into() }.into()
            );
        }
        if blocks_per_sm == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "blocks_per_sm must be > 0".into(),
            }
            .into());
        }
        if threads_per_block == 0 || threads_per_block > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: "threads_per_block must be 1..=1024".into(),
            }
            .into());
        }
        Ok(Self { sm_count, blocks_per_sm, threads_per_block, total_work_items })
    }

    /// Create from architecture defaults with occupancy-aware block count.
    pub fn from_arch(arch: &GpuArchitecture, total_work_items: u64) -> Self {
        let blocks_per_sm = (arch.max_blocks_per_sm / 4).max(1);
        Self { sm_count: arch.sm_count, blocks_per_sm, threads_per_block: 256, total_work_items }
    }

    /// Total persistent grid size.
    pub fn grid_size(&self) -> u32 {
        self.sm_count * self.blocks_per_sm
    }

    /// Total persistent thread count.
    pub fn total_threads(&self) -> u64 {
        self.grid_size() as u64 * self.threads_per_block as u64
    }

    /// Iterations per thread in the grid-stride loop.
    pub fn iterations_per_thread(&self) -> u64 {
        let total_threads = self.total_threads();
        if total_threads == 0 {
            return 0;
        }
        self.total_work_items.div_ceil(total_threads)
    }

    /// Convert to a [`LaunchConfig`].
    pub fn to_launch_config(&self) -> LaunchConfig {
        LaunchConfig::new_1d(self.grid_size(), self.threads_per_block)
    }
}

// ── KernelBatch / BatchBuilder ───────────────────────────────────────

static NEXT_BATCH_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a kernel batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(u64);

impl BatchId {
    fn next() -> Self {
        Self(NEXT_BATCH_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for BatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "batch-{}", self.0)
    }
}

/// A single kernel entry within a batch.
#[derive(Debug, Clone)]
pub struct BatchEntry {
    /// Kernel name.
    pub name: String,
    /// Launch configuration.
    pub config: LaunchConfig,
    /// Optional inter-kernel dependency (index into the batch).
    pub depends_on: Option<usize>,
}

/// Result of executing a kernel batch.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Batch identifier.
    pub id: BatchId,
    /// Number of kernels executed.
    pub kernels_launched: usize,
    /// Whether all kernels succeeded.
    pub success: bool,
    /// Per-kernel error messages (empty on success).
    pub errors: Vec<String>,
}

/// A batch of kernel launches that share a single synchronization point.
#[derive(Debug, Clone)]
pub struct KernelBatch {
    /// Batch identifier.
    pub id: BatchId,
    /// Ordered list of kernels.
    entries: Vec<BatchEntry>,
}

impl KernelBatch {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self { id: BatchId::next(), entries: Vec::new() }
    }

    /// Number of kernels in the batch.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a kernel to the batch.
    pub fn push(&mut self, entry: BatchEntry) {
        self.entries.push(entry);
    }

    /// Access entries.
    pub fn entries(&self) -> &[BatchEntry] {
        &self.entries
    }

    /// Execute the batch (CPU simulation: validates configs, counts launches).
    pub fn execute(&self) -> Result<BatchResult> {
        let mut errors = Vec::new();
        for (i, entry) in self.entries.iter().enumerate() {
            if let Some(dep) = entry.depends_on
                && dep >= i
            {
                errors.push(format!(
                    "kernel '{}' at index {i} depends on future index {dep}",
                    entry.name
                ));
            }
            if let Err(e) = LaunchValidator::validate(&entry.config, &GpuArchitecture::default()) {
                errors.push(format!("kernel '{}': {e}", entry.name));
            }
        }

        Ok(BatchResult {
            id: self.id,
            kernels_launched: if errors.is_empty() { self.entries.len() } else { 0 },
            success: errors.is_empty(),
            errors,
        })
    }
}

impl Default for KernelBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing kernel batches.
#[derive(Debug)]
pub struct BatchBuilder {
    entries: Vec<BatchEntry>,
}

impl BatchBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a kernel with no dependency.
    pub fn add(mut self, name: impl Into<String>, config: LaunchConfig) -> Self {
        self.entries.push(BatchEntry { name: name.into(), config, depends_on: None });
        self
    }

    /// Add a kernel that depends on a previous kernel (by index).
    pub fn add_dependent(
        mut self,
        name: impl Into<String>,
        config: LaunchConfig,
        depends_on: usize,
    ) -> Self {
        self.entries.push(BatchEntry { name: name.into(), config, depends_on: Some(depends_on) });
        self
    }

    /// Build the batch.
    pub fn build(self) -> KernelBatch {
        let mut batch = KernelBatch::new();
        for entry in self.entries {
            batch.push(entry);
        }
        batch
    }
}

impl Default for BatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── LaunchValidator ──────────────────────────────────────────────────

/// Validation errors for launch configurations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchValidationError {
    /// Short error code.
    pub code: &'static str,
    /// Human-readable description.
    pub message: String,
}

impl fmt::Display for LaunchValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

/// Validates kernel launch parameters against hardware limits.
pub struct LaunchValidator;

impl LaunchValidator {
    /// Validate a launch config against a GPU architecture.
    /// Returns all validation errors found.
    pub fn validate_all(
        config: &LaunchConfig,
        arch: &GpuArchitecture,
    ) -> Vec<LaunchValidationError> {
        let mut errors = Vec::new();

        // Block dimension checks.
        let tpb = config.threads_per_block();
        if tpb == 0 {
            errors.push(LaunchValidationError {
                code: "ZERO_BLOCK",
                message: "threads per block is 0".into(),
            });
        }
        if tpb > arch.max_threads_per_block {
            errors.push(LaunchValidationError {
                code: "BLOCK_TOO_LARGE",
                message: format!(
                    "threads per block ({tpb}) exceeds max ({})",
                    arch.max_threads_per_block
                ),
            });
        }
        for (i, dim_name) in ["x", "y", "z"].iter().enumerate() {
            if config.block[i] > arch.max_block_dim[i] {
                errors.push(LaunchValidationError {
                    code: "BLOCK_DIM_EXCEEDED",
                    message: format!(
                        "block.{dim_name}={} exceeds max {}",
                        config.block[i], arch.max_block_dim[i]
                    ),
                });
            }
        }

        // Grid dimension checks.
        for (i, dim_name) in ["x", "y", "z"].iter().enumerate() {
            if config.grid[i] == 0 {
                errors.push(LaunchValidationError {
                    code: "ZERO_GRID",
                    message: format!("grid.{dim_name} is 0"),
                });
            }
            if config.grid[i] > arch.max_grid_dim[i] {
                errors.push(LaunchValidationError {
                    code: "GRID_DIM_EXCEEDED",
                    message: format!(
                        "grid.{dim_name}={} exceeds max {}",
                        config.grid[i], arch.max_grid_dim[i]
                    ),
                });
            }
        }

        // Shared memory check.
        if config.shared_mem_bytes > arch.max_shared_mem_per_block {
            errors.push(LaunchValidationError {
                code: "SHARED_MEM_EXCEEDED",
                message: format!(
                    "shared memory {}B exceeds per-block limit {}B",
                    config.shared_mem_bytes, arch.max_shared_mem_per_block
                ),
            });
        }

        errors
    }

    /// Validate and return `Ok(())` or the first error.
    pub fn validate(config: &LaunchConfig, arch: &GpuArchitecture) -> Result<()> {
        let errors = Self::validate_all(config, arch);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(KernelError::InvalidArguments { reason: errors[0].to_string() }.into())
        }
    }

    /// Check that a block size is warp-aligned (multiple of 32).
    pub fn is_warp_aligned(threads_per_block: u32) -> bool {
        threads_per_block > 0 && threads_per_block.is_multiple_of(32)
    }

    /// Suggest the nearest warp-aligned block size (round up).
    pub fn align_to_warp(threads: u32) -> u32 {
        if threads == 0 {
            return 32;
        }
        threads.div_ceil(32) * 32
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── GpuArchitecture tests ────────────────────────────────────────

    #[test]
    fn arch_from_known_compute_capabilities() {
        let pascal = GpuArchitecture::from_compute_capability(6, 1, 28).unwrap();
        assert_eq!(pascal.family, GpuArchFamily::Pascal);
        assert_eq!(pascal.max_threads_per_sm, 2048);

        let volta = GpuArchitecture::from_compute_capability(7, 0, 80).unwrap();
        assert_eq!(volta.family, GpuArchFamily::Volta);

        let turing = GpuArchitecture::from_compute_capability(7, 5, 72).unwrap();
        assert_eq!(turing.family, GpuArchFamily::Turing);
        assert_eq!(turing.max_threads_per_sm, 1024);

        let ampere80 = GpuArchitecture::from_compute_capability(8, 0, 108).unwrap();
        assert_eq!(ampere80.family, GpuArchFamily::Ampere);
        assert_eq!(ampere80.max_shared_mem_per_sm, 163_840);

        let ampere86 = GpuArchitecture::from_compute_capability(8, 6, 84).unwrap();
        assert_eq!(ampere86.family, GpuArchFamily::Ampere);

        let ada = GpuArchitecture::from_compute_capability(8, 9, 128).unwrap();
        assert_eq!(ada.family, GpuArchFamily::Ada);

        let hopper = GpuArchitecture::from_compute_capability(9, 0, 132).unwrap();
        assert_eq!(hopper.family, GpuArchFamily::Hopper);
        assert_eq!(hopper.max_shared_mem_per_sm, 233_472);
    }

    #[test]
    fn arch_unknown_compute_capability() {
        assert!(GpuArchitecture::from_compute_capability(5, 0, 1).is_err());
        assert!(GpuArchitecture::from_compute_capability(10, 0, 1).is_err());
    }

    #[test]
    fn arch_default_is_ampere_a100() {
        let arch = GpuArchitecture::default();
        assert_eq!(arch.family, GpuArchFamily::Ampere);
        assert_eq!(arch.sm_count, 108);
        assert_eq!(arch.compute_capability, (8, 0));
    }

    #[test]
    fn arch_hopper_h100() {
        let arch = GpuArchitecture::hopper_h100();
        assert_eq!(arch.family, GpuArchFamily::Hopper);
        assert_eq!(arch.sm_count, 132);
    }

    #[test]
    fn arch_max_warps() {
        let a100 = GpuArchitecture::ampere_a100();
        assert_eq!(a100.max_warps_per_sm(), 64); // 2048 / 32

        let turing = GpuArchitecture::from_compute_capability(7, 5, 72).unwrap();
        assert_eq!(turing.max_warps_per_sm(), 32); // 1024 / 32
    }

    #[test]
    fn arch_feature_detection() {
        let pascal = GpuArchitecture::from_compute_capability(6, 1, 28).unwrap();
        assert!(pascal.has_native_fp16());
        assert!(!pascal.has_tensor_cores());
        assert!(!pascal.has_async_copy());

        let volta = GpuArchitecture::from_compute_capability(7, 0, 80).unwrap();
        assert!(volta.has_tensor_cores());
        assert!(!volta.has_async_copy());

        let ampere = GpuArchitecture::ampere_a100();
        assert!(ampere.has_tensor_cores());
        assert!(ampere.has_async_copy());
    }

    #[test]
    fn arch_display() {
        assert_eq!(GpuArchFamily::Pascal.to_string(), "Pascal (SM 6.x)");
        assert_eq!(GpuArchFamily::Hopper.to_string(), "Hopper (SM 9.0)");
    }

    #[test]
    fn arch_warp_size_is_32() {
        let arch = GpuArchitecture::default();
        assert_eq!(arch.warp_size, 32);
    }

    #[test]
    fn arch_max_block_and_grid_dims() {
        let arch = GpuArchitecture::default();
        assert_eq!(arch.max_threads_per_block, 1024);
        assert_eq!(arch.max_block_dim, [1024, 1024, 64]);
        assert_eq!(arch.max_grid_dim[0], 2_147_483_647);
    }

    // ── LaunchConfig tests ───────────────────────────────────────────

    #[test]
    fn launch_config_1d() {
        let cfg = LaunchConfig::new_1d(128, 256);
        assert_eq!(cfg.grid, [128, 1, 1]);
        assert_eq!(cfg.block, [256, 1, 1]);
        assert_eq!(cfg.shared_mem_bytes, 0);
        assert_eq!(cfg.threads_per_block(), 256);
        assert_eq!(cfg.total_blocks(), 128);
        assert_eq!(cfg.total_threads(), 128 * 256);
    }

    #[test]
    fn launch_config_2d() {
        let cfg = LaunchConfig::new_2d([32, 16], [16, 16]);
        assert_eq!(cfg.grid, [32, 16, 1]);
        assert_eq!(cfg.block, [16, 16, 1]);
        assert_eq!(cfg.threads_per_block(), 256);
        assert_eq!(cfg.total_blocks(), 512);
    }

    #[test]
    fn launch_config_3d() {
        let cfg = LaunchConfig::new_3d([4, 4, 4], [8, 8, 4]);
        assert_eq!(cfg.threads_per_block(), 256);
        assert_eq!(cfg.total_blocks(), 64);
        assert_eq!(cfg.total_threads(), 64 * 256);
    }

    #[test]
    fn launch_config_with_shared_mem() {
        let cfg = LaunchConfig::new_1d(10, 128).with_shared_mem(4096);
        assert_eq!(cfg.shared_mem_bytes, 4096);
    }

    #[test]
    fn launch_config_display() {
        let cfg = LaunchConfig::new_1d(4, 256).with_shared_mem(1024);
        let s = cfg.to_string();
        assert!(s.contains("4"));
        assert!(s.contains("256"));
        assert!(s.contains("1024"));
    }

    // ── OccupancyInfo / OccupancyLimiter tests ───────────────────────

    #[test]
    fn occupancy_limiter_display() {
        assert_eq!(OccupancyLimiter::Threads.to_string(), "threads");
        assert_eq!(OccupancyLimiter::Blocks.to_string(), "blocks");
        assert_eq!(OccupancyLimiter::SharedMemory.to_string(), "shared_memory");
        assert_eq!(OccupancyLimiter::Registers.to_string(), "registers");
    }

    // ── LaunchOptimizer tests ────────────────────────────────────────

    #[test]
    fn optimizer_1d_basic() {
        let arch = GpuArchitecture::ampere_a100();
        let mut opt = LaunchOptimizer::new(arch);
        let res = KernelResourceUsage::default();
        let cfg = opt.optimize_1d("elementwise", 1024, &res).unwrap();
        assert!(cfg.threads_per_block() > 0);
        assert!(cfg.total_threads() >= 1024);
    }

    #[test]
    fn optimizer_1d_large_problem() {
        let arch = GpuArchitecture::ampere_a100();
        let mut opt = LaunchOptimizer::new(arch);
        let res = KernelResourceUsage::default();
        let cfg = opt.optimize_1d("big", 10_000_000, &res).unwrap();
        assert!(cfg.total_threads() >= 10_000_000);
    }

    #[test]
    fn optimizer_1d_zero_problem_size_errors() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();
        assert!(opt.optimize_1d("bad", 0, &res).is_err());
    }

    #[test]
    fn optimizer_2d_basic() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage::default();
        let cfg = opt.optimize_2d("matmul", 512, 512, &res).unwrap();
        assert!(cfg.block[0] > 0);
        assert!(cfg.block[1] > 0);
        assert!(cfg.grid[0] > 0);
        assert!(cfg.grid[1] > 0);
    }

    #[test]
    fn optimizer_2d_zero_dim_errors() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();
        assert!(opt.optimize_2d("k", 0, 10, &res).is_err());
        assert!(opt.optimize_2d("k", 10, 0, &res).is_err());
    }

    #[test]
    fn optimizer_3d_basic() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage::default();
        let cfg = opt.optimize_3d("batched", [16, 64, 64], &res).unwrap();
        assert!(cfg.block[0] > 0);
        assert!(cfg.block[1] > 0);
        assert!(cfg.block[2] > 0);
    }

    #[test]
    fn optimizer_3d_zero_dim_errors() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();
        assert!(opt.optimize_3d("k", [0, 1, 1], &res).is_err());
        assert!(opt.optimize_3d("k", [1, 0, 1], &res).is_err());
        assert!(opt.optimize_3d("k", [1, 1, 0], &res).is_err());
    }

    #[test]
    fn optimizer_cache_hit() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage::default();

        let c1 = opt.optimize_1d("kern", 512, &res).unwrap();
        assert_eq!(opt.cache_misses(), 1);
        assert_eq!(opt.cache_hits(), 0);

        let c2 = opt.optimize_1d("kern", 512, &res).unwrap();
        assert_eq!(opt.cache_hits(), 1);
        assert_eq!(c1, c2);
    }

    #[test]
    fn optimizer_cache_miss_different_kernel() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();

        opt.optimize_1d("a", 100, &res).unwrap();
        opt.optimize_1d("b", 100, &res).unwrap();
        assert_eq!(opt.cache_misses(), 2);
    }

    #[test]
    fn optimizer_cache_miss_different_size() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();

        opt.optimize_1d("k", 100, &res).unwrap();
        opt.optimize_1d("k", 200, &res).unwrap();
        assert_eq!(opt.cache_misses(), 2);
    }

    #[test]
    fn optimizer_cache_clear() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();

        opt.optimize_1d("k", 100, &res).unwrap();
        assert_eq!(opt.cache_size(), 1);

        opt.clear_cache();
        assert_eq!(opt.cache_size(), 0);
        assert_eq!(opt.cache_hits(), 0);
        assert_eq!(opt.cache_misses(), 0);
    }

    #[test]
    fn optimizer_hit_rate() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        assert_eq!(opt.hit_rate(), 0.0);

        let res = KernelResourceUsage::default();
        opt.optimize_1d("k", 100, &res).unwrap(); // miss
        opt.optimize_1d("k", 100, &res).unwrap(); // hit
        opt.optimize_1d("k", 100, &res).unwrap(); // hit
        assert!((opt.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn optimizer_2d_cache_hit() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();

        let c1 = opt.optimize_2d("mm", 256, 256, &res).unwrap();
        let c2 = opt.optimize_2d("mm", 256, 256, &res).unwrap();
        assert_eq!(c1, c2);
        assert_eq!(opt.cache_hits(), 1);
    }

    #[test]
    fn optimizer_3d_cache_hit() {
        let mut opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();

        let c1 = opt.optimize_3d("b", [8, 8, 8], &res).unwrap();
        let c2 = opt.optimize_3d("b", [8, 8, 8], &res).unwrap();
        assert_eq!(c1, c2);
        assert_eq!(opt.cache_hits(), 1);
    }

    // ── Occupancy tests ──────────────────────────────────────────────

    #[test]
    fn occupancy_full_with_256_threads() {
        let opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage { registers_per_thread: 32, ..Default::default() };
        let occ = opt.compute_occupancy(256, &res);
        assert!(occ.occupancy > 0.0);
        assert!(occ.active_warps_per_sm > 0);
    }

    #[test]
    fn occupancy_zero_threads() {
        let opt = LaunchOptimizer::new(GpuArchitecture::default());
        let res = KernelResourceUsage::default();
        let occ = opt.compute_occupancy(0, &res);
        assert_eq!(occ.occupancy, 0.0);
        assert_eq!(occ.active_warps_per_sm, 0);
    }

    #[test]
    fn occupancy_limited_by_registers() {
        let opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage { registers_per_thread: 255, ..Default::default() };
        let occ = opt.compute_occupancy(256, &res);
        assert_eq!(occ.limiter, OccupancyLimiter::Registers);
    }

    #[test]
    fn occupancy_limited_by_shared_memory() {
        let opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage {
            registers_per_thread: 16,
            dynamic_shared_mem: 48_000,
            ..Default::default()
        };
        let occ = opt.compute_occupancy(256, &res);
        assert_eq!(occ.limiter, OccupancyLimiter::SharedMemory);
    }

    #[test]
    fn occupancy_1024_threads_block_limited() {
        let opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage { registers_per_thread: 16, ..Default::default() };
        let occ = opt.compute_occupancy(1024, &res);
        // 2048/1024 = 2 blocks by threads, vs 32 max blocks → limited by threads
        assert_eq!(occ.limiter, OccupancyLimiter::Threads);
        assert_eq!(occ.active_blocks_per_sm, 2);
    }

    #[test]
    fn occupancy_increases_with_smaller_blocks() {
        let opt = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let res = KernelResourceUsage::default();
        let occ_256 = opt.compute_occupancy(256, &res);
        let occ_1024 = opt.compute_occupancy(1024, &res);
        assert!(occ_256.occupancy >= occ_1024.occupancy);
    }

    #[test]
    fn occupancy_across_architectures() {
        let res = KernelResourceUsage::default();

        let ampere = LaunchOptimizer::new(GpuArchitecture::ampere_a100());
        let turing =
            LaunchOptimizer::new(GpuArchitecture::from_compute_capability(7, 5, 72).unwrap());

        let occ_a = ampere.compute_occupancy(256, &res);
        let occ_t = turing.compute_occupancy(256, &res);

        assert!(occ_a.occupancy > 0.0);
        assert!(occ_t.occupancy > 0.0);
        // Ampere has more warps/SM
        assert!(occ_a.max_warps_per_sm > occ_t.max_warps_per_sm);
    }

    // ── SharedMemoryPlanner tests ────────────────────────────────────

    #[test]
    fn shared_mem_none_strategy() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default());
        assert_eq!(planner.compute(SharedMemStrategy::None, 256).unwrap(), 0);
    }

    #[test]
    fn shared_mem_per_thread() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default());
        assert_eq!(planner.compute(SharedMemStrategy::PerThread, 256).unwrap(), 256 * 4);
    }

    #[test]
    fn shared_mem_per_thread_f16() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default()).with_elem_size(2);
        assert_eq!(planner.compute(SharedMemStrategy::PerThread, 256).unwrap(), 256 * 2);
    }

    #[test]
    fn shared_mem_fixed() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default());
        assert_eq!(planner.compute(SharedMemStrategy::Fixed(2048), 256).unwrap(), 2048);
    }

    #[test]
    fn shared_mem_per_warp() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default());
        // 256 threads / 32 = 8 warps * 4 bytes = 32
        assert_eq!(planner.compute(SharedMemStrategy::PerWarp, 256).unwrap(), 32);
    }

    #[test]
    fn shared_mem_tiled() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default());
        // 32 * 32 * 4 = 4096
        assert_eq!(
            planner
                .compute(SharedMemStrategy::Tiled { tile_rows: 32, tile_cols: 32 }, 256)
                .unwrap(),
            4096
        );
    }

    #[test]
    fn shared_mem_exceeds_limit() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default());
        // 49152 is max per block for Ampere; 50000 should fail
        assert!(planner.compute(SharedMemStrategy::Fixed(50_000), 256).is_err());
    }

    #[test]
    fn shared_mem_max_tile_size() {
        let planner = SharedMemoryPlanner::new(GpuArchitecture::default());
        let max = planner.max_tile_size();
        // sqrt(49152 / 4) = sqrt(12288) ≈ 110
        assert!(max > 100);
        assert!(max <= 111);
    }

    // ── ReductionLaunchPlanner tests ─────────────────────────────────

    #[test]
    fn reduction_single_element() {
        let planner = ReductionLaunchPlanner::default();
        let passes = planner.plan(1).unwrap();
        assert!(passes.is_empty());
    }

    #[test]
    fn reduction_small_input() {
        let planner = ReductionLaunchPlanner::default();
        let passes = planner.plan(100).unwrap();
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].block_size, 256);
        assert_eq!(passes[0].output_size, 1);
    }

    #[test]
    fn reduction_multi_pass() {
        let planner = ReductionLaunchPlanner::default();
        // 256 block * 4 ept = 1024 per block. 1M / 1024 = ~977 blocks first pass
        let passes = planner.plan(1_000_000).unwrap();
        assert!(passes.len() >= 2);
        assert_eq!(passes[0].input_size, 1_000_000);
        assert!(passes[0].output_size > 1);
        assert_eq!(passes.last().unwrap().output_size, 1);
    }

    #[test]
    fn reduction_progressive_shrinking() {
        let planner = ReductionLaunchPlanner::default();
        let passes = planner.plan(10_000_000).unwrap();
        for i in 1..passes.len() {
            assert!(passes[i].input_size < passes[i - 1].input_size);
        }
    }

    #[test]
    fn reduction_zero_elements_errors() {
        let planner = ReductionLaunchPlanner::default();
        assert!(planner.plan(0).is_err());
    }

    #[test]
    fn reduction_custom_block_size() {
        let planner = ReductionLaunchPlanner::new(512, 8).unwrap();
        let passes = planner.plan(100_000).unwrap();
        assert_eq!(passes[0].block_size, 512);
    }

    #[test]
    fn reduction_invalid_block_size() {
        assert!(ReductionLaunchPlanner::new(0, 4).is_err());
        assert!(ReductionLaunchPlanner::new(100, 4).is_err()); // not power of 2
        assert!(ReductionLaunchPlanner::new(256, 0).is_err());
    }

    #[test]
    fn reduction_pass_count() {
        let planner = ReductionLaunchPlanner::default();
        assert_eq!(planner.pass_count(1), 0);
        assert_eq!(planner.pass_count(100), 1);
        assert!(planner.pass_count(1_000_000) >= 2);
    }

    #[test]
    fn reduction_shared_mem_proportional_to_block() {
        let planner = ReductionLaunchPlanner::new(128, 4).unwrap();
        let passes = planner.plan(1000).unwrap();
        assert_eq!(passes[0].shared_mem_bytes, 128 * 4);

        let planner2 = ReductionLaunchPlanner::new(512, 4).unwrap();
        let passes2 = planner2.plan(1000).unwrap();
        assert_eq!(passes2[0].shared_mem_bytes, 512 * 4);
    }

    // ── PersistentKernelConfig tests ─────────────────────────────────

    #[test]
    fn persistent_basic() {
        let cfg = PersistentKernelConfig::new(108, 2, 256, 10_000_000).unwrap();
        assert_eq!(cfg.grid_size(), 216);
        assert_eq!(cfg.total_threads(), 216 * 256);
        assert!(cfg.iterations_per_thread() > 0);
    }

    #[test]
    fn persistent_from_arch() {
        let arch = GpuArchitecture::ampere_a100();
        let cfg = PersistentKernelConfig::from_arch(&arch, 1_000_000);
        assert_eq!(cfg.sm_count, 108);
        assert!(cfg.grid_size() > 0);
    }

    #[test]
    fn persistent_iterations_cover_work() {
        let cfg = PersistentKernelConfig::new(10, 2, 128, 50_000).unwrap();
        let total = cfg.total_threads() * cfg.iterations_per_thread();
        assert!(total >= 50_000);
    }

    #[test]
    fn persistent_to_launch_config() {
        let cfg = PersistentKernelConfig::new(10, 2, 256, 10_000).unwrap();
        let lc = cfg.to_launch_config();
        assert_eq!(lc.grid, [20, 1, 1]);
        assert_eq!(lc.block, [256, 1, 1]);
    }

    #[test]
    fn persistent_zero_sm_errors() {
        assert!(PersistentKernelConfig::new(0, 2, 256, 100).is_err());
    }

    #[test]
    fn persistent_zero_blocks_per_sm_errors() {
        assert!(PersistentKernelConfig::new(10, 0, 256, 100).is_err());
    }

    #[test]
    fn persistent_invalid_thread_count() {
        assert!(PersistentKernelConfig::new(10, 2, 0, 100).is_err());
        assert!(PersistentKernelConfig::new(10, 2, 2048, 100).is_err());
    }

    #[test]
    fn persistent_zero_work_items() {
        let cfg = PersistentKernelConfig::new(10, 2, 256, 0).unwrap();
        assert_eq!(cfg.iterations_per_thread(), 0);
    }

    // ── KernelBatch / BatchBuilder tests ─────────────────────────────

    #[test]
    fn batch_empty() {
        let batch = KernelBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn batch_push_and_len() {
        let mut batch = KernelBatch::new();
        batch.push(BatchEntry {
            name: "k1".into(),
            config: LaunchConfig::new_1d(4, 256),
            depends_on: None,
        });
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn batch_execute_success() {
        let batch = BatchBuilder::new()
            .add("k1", LaunchConfig::new_1d(4, 256))
            .add("k2", LaunchConfig::new_1d(8, 128))
            .build();
        let result = batch.execute().unwrap();
        assert!(result.success);
        assert_eq!(result.kernels_launched, 2);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn batch_execute_with_dependency() {
        let batch = BatchBuilder::new()
            .add("k1", LaunchConfig::new_1d(4, 256))
            .add_dependent("k2", LaunchConfig::new_1d(4, 256), 0)
            .build();
        let result = batch.execute().unwrap();
        assert!(result.success);
        assert_eq!(result.kernels_launched, 2);
    }

    #[test]
    fn batch_execute_forward_dependency_error() {
        let batch = BatchBuilder::new()
            .add_dependent("k1", LaunchConfig::new_1d(4, 256), 1)
            .add("k2", LaunchConfig::new_1d(4, 256))
            .build();
        let result = batch.execute().unwrap();
        assert!(!result.success);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn batch_execute_invalid_config() {
        let bad = LaunchConfig::new_1d(0, 256); // grid 0
        let batch = BatchBuilder::new().add("bad", bad).build();
        let result = batch.execute().unwrap();
        assert!(!result.success);
    }

    #[test]
    fn batch_builder_default() {
        let builder = BatchBuilder::default();
        let batch = builder.build();
        assert!(batch.is_empty());
    }

    #[test]
    fn batch_default() {
        let batch = KernelBatch::default();
        assert!(batch.is_empty());
    }

    #[test]
    fn batch_entries_accessible() {
        let batch = BatchBuilder::new()
            .add("a", LaunchConfig::new_1d(1, 32))
            .add("b", LaunchConfig::new_1d(2, 64))
            .build();
        assert_eq!(batch.entries().len(), 2);
        assert_eq!(batch.entries()[0].name, "a");
        assert_eq!(batch.entries()[1].name, "b");
    }

    #[test]
    fn batch_id_unique() {
        let b1 = KernelBatch::new();
        let b2 = KernelBatch::new();
        assert_ne!(b1.id, b2.id);
    }

    #[test]
    fn batch_id_display() {
        let batch = KernelBatch::new();
        let s = batch.id.to_string();
        assert!(s.starts_with("batch-"));
    }

    // ── LaunchValidator tests ────────────────────────────────────────

    #[test]
    fn validator_valid_config() {
        let cfg = LaunchConfig::new_1d(128, 256);
        let arch = GpuArchitecture::default();
        assert!(LaunchValidator::validate(&cfg, &arch).is_ok());
    }

    #[test]
    fn validator_zero_block_threads() {
        let cfg = LaunchConfig::new_1d(1, 0);
        let arch = GpuArchitecture::default();
        let errs = LaunchValidator::validate_all(&cfg, &arch);
        assert!(errs.iter().any(|e| e.code == "ZERO_BLOCK"));
    }

    #[test]
    fn validator_block_too_large() {
        let cfg = LaunchConfig { grid: [1, 1, 1], block: [2048, 1, 1], shared_mem_bytes: 0 };
        let arch = GpuArchitecture::default();
        let errs = LaunchValidator::validate_all(&cfg, &arch);
        assert!(errs.iter().any(|e| e.code == "BLOCK_TOO_LARGE"));
    }

    #[test]
    fn validator_block_dim_exceeded_z() {
        let cfg = LaunchConfig {
            grid: [1, 1, 1],
            block: [1, 1, 128], // max z is 64
            shared_mem_bytes: 0,
        };
        let arch = GpuArchitecture::default();
        let errs = LaunchValidator::validate_all(&cfg, &arch);
        assert!(errs.iter().any(|e| e.code == "BLOCK_DIM_EXCEEDED"));
    }

    #[test]
    fn validator_zero_grid() {
        let cfg = LaunchConfig { grid: [0, 1, 1], block: [256, 1, 1], shared_mem_bytes: 0 };
        let arch = GpuArchitecture::default();
        let errs = LaunchValidator::validate_all(&cfg, &arch);
        assert!(errs.iter().any(|e| e.code == "ZERO_GRID"));
    }

    #[test]
    fn validator_shared_mem_exceeded() {
        let cfg = LaunchConfig::new_1d(1, 256).with_shared_mem(100_000);
        let arch = GpuArchitecture::default();
        let errs = LaunchValidator::validate_all(&cfg, &arch);
        assert!(errs.iter().any(|e| e.code == "SHARED_MEM_EXCEEDED"));
    }

    #[test]
    fn validator_multiple_errors() {
        let cfg = LaunchConfig { grid: [0, 1, 1], block: [0, 1, 1], shared_mem_bytes: 100_000 };
        let arch = GpuArchitecture::default();
        let errs = LaunchValidator::validate_all(&cfg, &arch);
        assert!(errs.len() >= 3);
    }

    #[test]
    fn validator_validate_returns_first_error() {
        let cfg = LaunchConfig { grid: [0, 1, 1], block: [0, 1, 1], shared_mem_bytes: 0 };
        let arch = GpuArchitecture::default();
        assert!(LaunchValidator::validate(&cfg, &arch).is_err());
    }

    #[test]
    fn validator_warp_aligned() {
        assert!(LaunchValidator::is_warp_aligned(32));
        assert!(LaunchValidator::is_warp_aligned(256));
        assert!(LaunchValidator::is_warp_aligned(1024));
        assert!(!LaunchValidator::is_warp_aligned(0));
        assert!(!LaunchValidator::is_warp_aligned(100));
        assert!(!LaunchValidator::is_warp_aligned(33));
    }

    #[test]
    fn validator_align_to_warp() {
        assert_eq!(LaunchValidator::align_to_warp(0), 32);
        assert_eq!(LaunchValidator::align_to_warp(1), 32);
        assert_eq!(LaunchValidator::align_to_warp(32), 32);
        assert_eq!(LaunchValidator::align_to_warp(33), 64);
        assert_eq!(LaunchValidator::align_to_warp(256), 256);
        assert_eq!(LaunchValidator::align_to_warp(257), 288);
    }

    #[test]
    fn validation_error_display() {
        let err = LaunchValidationError { code: "TEST", message: "test message".into() };
        assert_eq!(err.to_string(), "[TEST] test message");
    }

    // ── Factor block dimension tests ─────────────────────────────────

    #[test]
    fn factor_2d_256() {
        let (bx, by) = LaunchOptimizer::factor_block_2d(256);
        assert_eq!(bx * by, 256);
        assert!(bx <= 1024);
        assert!(by <= 1024);
    }

    #[test]
    fn factor_2d_1024() {
        let (bx, by) = LaunchOptimizer::factor_block_2d(1024);
        assert_eq!(bx * by, 1024);
    }

    #[test]
    fn factor_2d_32() {
        let (bx, by) = LaunchOptimizer::factor_block_2d(32);
        assert_eq!(bx * by, 32);
    }

    #[test]
    fn factor_3d_produces_valid_block() {
        let (bx, by, bz) = LaunchOptimizer::factor_block_3d(256);
        assert_eq!(bx * by * bz, 256);
        assert!(bz >= 1);
    }

    // ── Integration / cross-cutting tests ────────────────────────────

    #[test]
    fn optimizer_respects_grid_dim_limits() {
        let arch = GpuArchitecture::from_compute_capability(6, 1, 10).unwrap();
        let mut opt = LaunchOptimizer::new(arch.clone());
        let res = KernelResourceUsage::default();
        // Even with huge N, grid should not exceed max.
        let cfg = opt.optimize_1d("k", u64::MAX / 2, &res).unwrap();
        assert!(cfg.grid[0] <= arch.max_grid_dim[0]);
    }

    #[test]
    fn optimizer_shared_mem_integration() {
        let arch = GpuArchitecture::ampere_a100();
        let planner = SharedMemoryPlanner::new(arch.clone());
        let smem = planner.compute(SharedMemStrategy::PerThread, 256).unwrap();

        let mut opt = LaunchOptimizer::new(arch);
        let res = KernelResourceUsage { dynamic_shared_mem: smem, ..Default::default() };
        let cfg = opt.optimize_1d("with_smem", 10_000, &res).unwrap();
        assert_eq!(cfg.shared_mem_bytes, smem);
    }

    #[test]
    fn persistent_launch_covers_all_work() {
        let arch = GpuArchitecture::hopper_h100();
        let total = 5_000_000u64;
        let cfg = PersistentKernelConfig::from_arch(&arch, total);
        let covered = cfg.total_threads() * cfg.iterations_per_thread();
        assert!(covered >= total);
    }

    #[test]
    fn reduction_then_validate() {
        let planner = ReductionLaunchPlanner::default();
        let arch = GpuArchitecture::default();
        let passes = planner.plan(100_000).unwrap();
        for pass in &passes {
            let cfg = LaunchConfig::new_1d(pass.grid_size, pass.block_size)
                .with_shared_mem(pass.shared_mem_bytes);
            assert!(LaunchValidator::validate(&cfg, &arch).is_ok());
        }
    }

    #[test]
    fn batch_with_optimized_configs() {
        let arch = GpuArchitecture::ampere_a100();
        let mut opt = LaunchOptimizer::new(arch);
        let res = KernelResourceUsage::default();

        let c1 = opt.optimize_1d("layernorm", 4096, &res).unwrap();
        let c2 = opt.optimize_1d("ffn", 4096 * 11008, &res).unwrap();

        let batch = BatchBuilder::new().add("layernorm", c1).add_dependent("ffn", c2, 0).build();
        let result = batch.execute().unwrap();
        assert!(result.success);
        assert_eq!(result.kernels_launched, 2);
    }

    #[test]
    fn resource_usage_default() {
        let res = KernelResourceUsage::default();
        assert_eq!(res.registers_per_thread, 32);
        assert_eq!(res.static_shared_mem, 0);
        assert_eq!(res.dynamic_shared_mem, 0);
    }

    #[test]
    fn optimizer_architecture_accessor() {
        let arch = GpuArchitecture::hopper_h100();
        let opt = LaunchOptimizer::new(arch);
        assert_eq!(opt.architecture().family, GpuArchFamily::Hopper);
    }
}
