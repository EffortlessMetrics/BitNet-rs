//! CUDA kernel launch configuration utilities.
//!
//! Provides types and functions for computing grid/block dimensions,
//! validating launch parameters against device limits, and selecting
//! optimal block sizes for 1-D, 2-D, and 3-D kernel launches.

use std::fmt;

// ── Constants ────────────────────────────────────────────────────────

/// Maximum grid dimension on any single axis (CUDA spec).
pub const MAX_GRID_DIM_X: u32 = 2_147_483_647; // 2^31 - 1
pub const MAX_GRID_DIM_YZ: u32 = 65_535;

/// Default warp size on all current NVIDIA architectures.
pub const WARP_SIZE: u32 = 32;

// ── Error type ───────────────────────────────────────────────────────

/// Errors that can occur when building or validating a launch config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    /// Total threads per block exceeds device maximum.
    ExceedsMaxThreadsPerBlock { requested: u32, limit: u32 },
    /// A grid dimension exceeds the device maximum.
    ExceedsMaxGridDim { axis: &'static str, requested: u32, limit: u32 },
    /// Shared memory request exceeds device capacity.
    ExceedsSharedMemory { requested: usize, limit: usize },
    /// A block dimension is zero.
    ZeroBlockDim { axis: &'static str },
    /// Total work size is zero.
    ZeroWorkSize,
    /// Block dimension is not a multiple of warp size.
    NotWarpAligned { block_dim_x: u32, warp_size: u32 },
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExceedsMaxThreadsPerBlock { requested, limit } => {
                write!(f, "threads per block {requested} exceeds limit {limit}")
            }
            Self::ExceedsMaxGridDim { axis, requested, limit } => {
                write!(f, "grid dim {axis}={requested} exceeds limit {limit}")
            }
            Self::ExceedsSharedMemory { requested, limit } => {
                write!(f, "shared memory {requested} B exceeds limit {limit} B")
            }
            Self::ZeroBlockDim { axis } => {
                write!(f, "block dim {axis} is zero")
            }
            Self::ZeroWorkSize => write!(f, "total work size is zero"),
            Self::NotWarpAligned { block_dim_x, warp_size } => {
                write!(
                    f,
                    "block_dim.x={block_dim_x} not aligned to warp \
                     size {warp_size}"
                )
            }
        }
    }
}

impl std::error::Error for LaunchError {}

// ── Core types ───────────────────────────────────────────────────────

/// Full CUDA launch configuration (grid, block, shared memory, stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: usize,
    pub stream_id: Option<u32>,
}

impl LaunchConfig {
    /// Total number of threads this configuration will launch.
    pub fn total_threads(&self) -> u64 {
        let grid = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let block = self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        grid * block
    }

    /// Threads per block as a single number.
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }
}

/// Compiler-directed launch bounds (mirrors `__launch_bounds__`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchBounds {
    pub max_threads_per_block: u32,
    pub min_blocks_per_sm: u32,
}

/// Describes how work is distributed across the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkDistribution {
    pub total_elements: usize,
    pub threads_per_block: u32,
    pub blocks: u32,
}

impl WorkDistribution {
    /// Number of "wasted" threads (launched but beyond `total_elements`).
    pub fn excess_threads(&self) -> usize {
        let launched = self.threads_per_block as usize * self.blocks as usize;
        launched.saturating_sub(self.total_elements)
    }
}

// ── Device limits ────────────────────────────────────────────────────

/// Hardware limits of a specific CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    pub max_threads_per_block: u32,
    pub max_grid_dims: (u32, u32, u32),
    pub max_shared_memory: usize,
    pub warp_size: u32,
    pub max_warps_per_sm: u32,
    pub sm_count: u32,
}

impl DeviceLimits {
    /// NVIDIA RTX 5070 Ti (SM 8.9, Blackwell-consumer).
    pub fn rtx_5070ti() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_grid_dims: (MAX_GRID_DIM_X, MAX_GRID_DIM_YZ, MAX_GRID_DIM_YZ),
            max_shared_memory: 100 * 1024, // 100 KiB
            warp_size: WARP_SIZE,
            max_warps_per_sm: 48,
            sm_count: 70,
        }
    }

    /// NVIDIA A100 (SM 8.0, Ampere data-centre).
    pub fn a100() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_grid_dims: (MAX_GRID_DIM_X, MAX_GRID_DIM_YZ, MAX_GRID_DIM_YZ),
            max_shared_memory: 164 * 1024, // 164 KiB configurable
            warp_size: WARP_SIZE,
            max_warps_per_sm: 64,
            sm_count: 108,
        }
    }

    /// Generic SM 8.9 (Ada Lovelace / Blackwell consumer).
    pub fn generic_sm89() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_grid_dims: (MAX_GRID_DIM_X, MAX_GRID_DIM_YZ, MAX_GRID_DIM_YZ),
            max_shared_memory: 100 * 1024,
            warp_size: WARP_SIZE,
            max_warps_per_sm: 48,
            sm_count: 128,
        }
    }

    /// Conservative limits that should work on any sm_50+ device.
    pub fn conservative() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_grid_dims: (MAX_GRID_DIM_X, MAX_GRID_DIM_YZ, MAX_GRID_DIM_YZ),
            max_shared_memory: 48 * 1024,
            warp_size: WARP_SIZE,
            max_warps_per_sm: 32,
            sm_count: 1,
        }
    }
}

impl Default for DeviceLimits {
    fn default() -> Self {
        Self::conservative()
    }
}

// ── Public API ───────────────────────────────────────────────────────

/// Compute a 1-D launch configuration.
///
/// `blocks = ceil(total_elements / threads_per_block)`.
pub fn compute_launch_config(total_elements: usize, threads_per_block: u32) -> LaunchConfig {
    assert!(threads_per_block > 0, "threads_per_block must be > 0");
    let blocks = div_ceil(total_elements, threads_per_block as usize) as u32;
    LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
        stream_id: None,
    }
}

/// Convenience wrapper: 1-D launch with explicit block size.
pub fn compute_1d_launch(n: usize, block_size: u32) -> LaunchConfig {
    compute_launch_config(n, block_size)
}

/// 2-D launch: each block is `(block_x, block_y)` threads.
pub fn compute_2d_launch(rows: usize, cols: usize, block_x: u32, block_y: u32) -> LaunchConfig {
    assert!(block_x > 0 && block_y > 0, "block dims must be > 0");
    let grid_x = div_ceil(cols, block_x as usize) as u32;
    let grid_y = div_ceil(rows, block_y as usize) as u32;
    LaunchConfig {
        grid_dim: (grid_x.max(1), grid_y.max(1), 1),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
        stream_id: None,
    }
}

/// 3-D launch.
pub fn compute_3d_launch(
    x: usize,
    y: usize,
    z: usize,
    block_dims: (u32, u32, u32),
) -> LaunchConfig {
    assert!(block_dims.0 > 0 && block_dims.1 > 0 && block_dims.2 > 0, "block dims must be > 0");
    let gx = div_ceil(x, block_dims.0 as usize) as u32;
    let gy = div_ceil(y, block_dims.1 as usize) as u32;
    let gz = div_ceil(z, block_dims.2 as usize) as u32;
    LaunchConfig {
        grid_dim: (gx.max(1), gy.max(1), gz.max(1)),
        block_dim: block_dims,
        shared_mem_bytes: 0,
        stream_id: None,
    }
}

/// Choose a block size that maximises occupancy heuristically.
///
/// Returns a warp-aligned power-of-two block size in `[32, 1024]`.
pub fn optimal_block_size(elements: usize, shared_mem_per_thread: usize) -> u32 {
    // Heuristic: start from 256 threads (good default), step down if
    // shared memory pressure is high, step down further for tiny problems.
    let max_candidate = if shared_mem_per_thread > 0 {
        // Rough cap so shared mem stays under 48 KiB (conservative).
        let cap = 48 * 1024 / shared_mem_per_thread.max(1);
        next_pow2_down(cap as u32).min(1024)
    } else {
        1024
    };

    // For small problems, don't over-launch.
    let size_cap = next_pow2_up(elements as u32).min(max_candidate);

    // Clamp to warp size.
    clamp_to_warp(size_cap)
}

/// Validate a `LaunchConfig` against concrete `DeviceLimits`.
pub fn validate_launch_config(
    config: &LaunchConfig,
    limits: &DeviceLimits,
) -> Result<(), LaunchError> {
    // Block dims must be non-zero.
    if config.block_dim.0 == 0 {
        return Err(LaunchError::ZeroBlockDim { axis: "x" });
    }
    if config.block_dim.1 == 0 {
        return Err(LaunchError::ZeroBlockDim { axis: "y" });
    }
    if config.block_dim.2 == 0 {
        return Err(LaunchError::ZeroBlockDim { axis: "z" });
    }

    let tpb = config.threads_per_block();
    if tpb > limits.max_threads_per_block {
        return Err(LaunchError::ExceedsMaxThreadsPerBlock {
            requested: tpb,
            limit: limits.max_threads_per_block,
        });
    }

    // Grid dim checks.
    if config.grid_dim.0 > limits.max_grid_dims.0 {
        return Err(LaunchError::ExceedsMaxGridDim {
            axis: "x",
            requested: config.grid_dim.0,
            limit: limits.max_grid_dims.0,
        });
    }
    if config.grid_dim.1 > limits.max_grid_dims.1 {
        return Err(LaunchError::ExceedsMaxGridDim {
            axis: "y",
            requested: config.grid_dim.1,
            limit: limits.max_grid_dims.1,
        });
    }
    if config.grid_dim.2 > limits.max_grid_dims.2 {
        return Err(LaunchError::ExceedsMaxGridDim {
            axis: "z",
            requested: config.grid_dim.2,
            limit: limits.max_grid_dims.2,
        });
    }

    // Shared memory.
    if config.shared_mem_bytes > limits.max_shared_memory {
        return Err(LaunchError::ExceedsSharedMemory {
            requested: config.shared_mem_bytes,
            limit: limits.max_shared_memory,
        });
    }

    Ok(())
}

/// Build a `WorkDistribution` for a 1-D problem.
pub fn work_distribution(total_elements: usize, threads_per_block: u32) -> WorkDistribution {
    let blocks = div_ceil(total_elements, threads_per_block as usize) as u32;
    WorkDistribution { total_elements, threads_per_block, blocks: blocks.max(1) }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Round *down* to the nearest power of two (≥ 1).
fn next_pow2_down(v: u32) -> u32 {
    if v == 0 {
        return 1;
    }
    1 << (31 - v.leading_zeros())
}

/// Round *up* to the nearest power of two (≥ 1).
fn next_pow2_up(v: u32) -> u32 {
    if v <= 1 {
        return 1;
    }
    1u32.checked_shl(32 - (v - 1).leading_zeros()).unwrap_or(1 << 31)
}

/// Clamp to `[WARP_SIZE, 1024]` and round down to warp multiple.
fn clamp_to_warp(v: u32) -> u32 {
    let v = v.clamp(WARP_SIZE, 1024);
    (v / WARP_SIZE) * WARP_SIZE
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- LaunchConfig basic ----------------------------------------

    #[test]
    fn test_launch_config_total_threads_1d() {
        let cfg = compute_launch_config(1024, 256);
        assert_eq!(cfg.total_threads(), 1024);
    }

    #[test]
    fn test_launch_config_threads_per_block() {
        let cfg = compute_launch_config(512, 128);
        assert_eq!(cfg.threads_per_block(), 128);
    }

    #[test]
    fn test_launch_config_stream_none() {
        let cfg = compute_launch_config(64, 32);
        assert_eq!(cfg.stream_id, None);
    }

    #[test]
    fn test_launch_config_shared_mem_default_zero() {
        let cfg = compute_launch_config(128, 64);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    // -- compute_launch_config / 1-D --------------------------------

    #[test]
    fn test_1d_exact_multiple() {
        let cfg = compute_launch_config(256, 256);
        assert_eq!(cfg.grid_dim, (1, 1, 1));
        assert_eq!(cfg.block_dim, (256, 1, 1));
    }

    #[test]
    fn test_1d_non_exact_rounds_up() {
        let cfg = compute_launch_config(257, 256);
        assert_eq!(cfg.grid_dim, (2, 1, 1));
    }

    #[test]
    fn test_1d_single_element() {
        let cfg = compute_launch_config(1, 32);
        assert_eq!(cfg.grid_dim, (1, 1, 1));
    }

    #[test]
    fn test_1d_large_problem() {
        let cfg = compute_launch_config(1_000_000, 512);
        assert_eq!(cfg.grid_dim.0, 1954); // ceil(1M/512)
    }

    #[test]
    fn test_compute_1d_launch_alias() {
        let a = compute_launch_config(500, 128);
        let b = compute_1d_launch(500, 128);
        assert_eq!(a, b);
    }

    // -- compute_2d_launch ------------------------------------------

    #[test]
    fn test_2d_exact_tiles() {
        let cfg = compute_2d_launch(32, 64, 16, 16);
        assert_eq!(cfg.grid_dim, (4, 2, 1));
        assert_eq!(cfg.block_dim, (16, 16, 1));
    }

    #[test]
    fn test_2d_non_exact_tiles() {
        let cfg = compute_2d_launch(33, 65, 16, 16);
        assert_eq!(cfg.grid_dim, (5, 3, 1));
    }

    #[test]
    fn test_2d_single_pixel() {
        let cfg = compute_2d_launch(1, 1, 16, 16);
        assert_eq!(cfg.grid_dim, (1, 1, 1));
    }

    #[test]
    fn test_2d_wide_image() {
        let cfg = compute_2d_launch(1, 4096, 32, 1);
        assert_eq!(cfg.grid_dim, (128, 1, 1));
        assert_eq!(cfg.block_dim, (32, 1, 1));
    }

    #[test]
    fn test_2d_tall_image() {
        let cfg = compute_2d_launch(4096, 1, 1, 32);
        assert_eq!(cfg.grid_dim, (1, 128, 1));
    }

    // -- compute_3d_launch ------------------------------------------

    #[test]
    fn test_3d_exact() {
        let cfg = compute_3d_launch(8, 8, 8, (8, 8, 8));
        assert_eq!(cfg.grid_dim, (1, 1, 1));
    }

    #[test]
    fn test_3d_non_exact() {
        let cfg = compute_3d_launch(9, 9, 9, (8, 8, 8));
        assert_eq!(cfg.grid_dim, (2, 2, 2));
    }

    #[test]
    fn test_3d_flat_z() {
        let cfg = compute_3d_launch(256, 256, 1, (16, 16, 1));
        assert_eq!(cfg.grid_dim, (16, 16, 1));
    }

    #[test]
    fn test_3d_block_dims_preserved() {
        let cfg = compute_3d_launch(10, 20, 30, (4, 8, 16));
        assert_eq!(cfg.block_dim, (4, 8, 16));
    }

    // -- optimal_block_size -----------------------------------------

    #[test]
    fn test_optimal_no_shared_mem_large() {
        let bs = optimal_block_size(1_000_000, 0);
        assert!(bs >= WARP_SIZE);
        assert!(bs <= 1024);
        assert_eq!(bs % WARP_SIZE, 0);
    }

    #[test]
    fn test_optimal_no_shared_mem_small() {
        let bs = optimal_block_size(16, 0);
        assert_eq!(bs, WARP_SIZE); // clamped to warp size
    }

    #[test]
    fn test_optimal_high_shared_mem() {
        // 1 KiB per thread → only ~48 threads fit in 48 KiB.
        let bs = optimal_block_size(10_000, 1024);
        assert!(bs <= 64);
        assert!(bs >= WARP_SIZE);
    }

    #[test]
    fn test_optimal_is_warp_aligned() {
        for elems in [1, 7, 33, 100, 500, 10_000, 1_000_000] {
            let bs = optimal_block_size(elems, 0);
            assert_eq!(bs % WARP_SIZE, 0, "elems={elems}");
        }
    }

    #[test]
    fn test_optimal_moderate_shared_mem() {
        let bs = optimal_block_size(50_000, 64);
        assert!(bs >= WARP_SIZE);
        assert!(bs <= 1024);
    }

    // -- validate_launch_config -------------------------------------

    #[test]
    fn test_validate_ok() {
        let cfg = compute_launch_config(1024, 256);
        let limits = DeviceLimits::a100();
        assert!(validate_launch_config(&cfg, &limits).is_ok());
    }

    #[test]
    fn test_validate_exceeds_threads_per_block() {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (2048, 1, 1),
            shared_mem_bytes: 0,
            stream_id: None,
        };
        let limits = DeviceLimits::a100();
        let err = validate_launch_config(&cfg, &limits).unwrap_err();
        assert_eq!(err, LaunchError::ExceedsMaxThreadsPerBlock { requested: 2048, limit: 1024 });
    }

    #[test]
    fn test_validate_exceeds_grid_y() {
        let cfg = LaunchConfig {
            grid_dim: (1, 70_000, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
            stream_id: None,
        };
        let limits = DeviceLimits::a100();
        let err = validate_launch_config(&cfg, &limits).unwrap_err();
        assert!(matches!(err, LaunchError::ExceedsMaxGridDim { axis: "y", .. }));
    }

    #[test]
    fn test_validate_exceeds_grid_z() {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 70_000),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
            stream_id: None,
        };
        let limits = DeviceLimits::a100();
        assert!(matches!(
            validate_launch_config(&cfg, &limits).unwrap_err(),
            LaunchError::ExceedsMaxGridDim { axis: "z", .. }
        ));
    }

    #[test]
    fn test_validate_exceeds_shared_memory() {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 200 * 1024,
            stream_id: None,
        };
        let limits = DeviceLimits::a100();
        assert!(matches!(
            validate_launch_config(&cfg, &limits).unwrap_err(),
            LaunchError::ExceedsSharedMemory { .. }
        ));
    }

    #[test]
    fn test_validate_zero_block_x() {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (0, 1, 1),
            shared_mem_bytes: 0,
            stream_id: None,
        };
        assert!(matches!(
            validate_launch_config(&cfg, &DeviceLimits::default()).unwrap_err(),
            LaunchError::ZeroBlockDim { axis: "x" }
        ));
    }

    #[test]
    fn test_validate_zero_block_y() {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 0, 1),
            shared_mem_bytes: 0,
            stream_id: None,
        };
        assert!(matches!(
            validate_launch_config(&cfg, &DeviceLimits::default()).unwrap_err(),
            LaunchError::ZeroBlockDim { axis: "y" }
        ));
    }

    #[test]
    fn test_validate_zero_block_z() {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 0),
            shared_mem_bytes: 0,
            stream_id: None,
        };
        assert!(matches!(
            validate_launch_config(&cfg, &DeviceLimits::default()).unwrap_err(),
            LaunchError::ZeroBlockDim { axis: "z" }
        ));
    }

    #[test]
    fn test_validate_at_exact_limits() {
        let limits = DeviceLimits {
            max_threads_per_block: 256,
            max_grid_dims: (100, 100, 100),
            max_shared_memory: 1024,
            warp_size: 32,
            max_warps_per_sm: 32,
            sm_count: 1,
        };
        let cfg = LaunchConfig {
            grid_dim: (100, 100, 100),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 1024,
            stream_id: None,
        };
        assert!(validate_launch_config(&cfg, &limits).is_ok());
    }

    // -- DeviceLimits presets ---------------------------------------

    #[test]
    fn test_a100_preset_sane() {
        let a = DeviceLimits::a100();
        assert_eq!(a.max_threads_per_block, 1024);
        assert_eq!(a.sm_count, 108);
        assert_eq!(a.warp_size, WARP_SIZE);
    }

    #[test]
    fn test_rtx_5070ti_preset_sane() {
        let r = DeviceLimits::rtx_5070ti();
        assert_eq!(r.max_threads_per_block, 1024);
        assert_eq!(r.sm_count, 70);
    }

    #[test]
    fn test_generic_sm89_preset_sane() {
        let g = DeviceLimits::generic_sm89();
        assert_eq!(g.sm_count, 128);
        assert_eq!(g.max_warps_per_sm, 48);
    }

    #[test]
    fn test_conservative_preset_sane() {
        let c = DeviceLimits::conservative();
        assert_eq!(c.sm_count, 1);
        assert!(c.max_shared_memory >= 48 * 1024);
    }

    #[test]
    fn test_default_is_conservative() {
        assert_eq!(DeviceLimits::default(), DeviceLimits::conservative());
    }

    // -- WorkDistribution -------------------------------------------

    #[test]
    fn test_work_distribution_exact() {
        let wd = work_distribution(1024, 256);
        assert_eq!(wd.blocks, 4);
        assert_eq!(wd.excess_threads(), 0);
    }

    #[test]
    fn test_work_distribution_with_excess() {
        let wd = work_distribution(1000, 256);
        assert_eq!(wd.blocks, 4); // 4 * 256 = 1024
        assert_eq!(wd.excess_threads(), 24);
    }

    #[test]
    fn test_work_distribution_single_element() {
        let wd = work_distribution(1, 256);
        assert_eq!(wd.blocks, 1);
        assert_eq!(wd.excess_threads(), 255);
    }

    // -- LaunchBounds -----------------------------------------------

    #[test]
    fn test_launch_bounds_construction() {
        let lb = LaunchBounds { max_threads_per_block: 256, min_blocks_per_sm: 2 };
        assert_eq!(lb.max_threads_per_block, 256);
        assert_eq!(lb.min_blocks_per_sm, 2);
    }

    // -- Error display ----------------------------------------------

    #[test]
    fn test_error_display_threads() {
        let e = LaunchError::ExceedsMaxThreadsPerBlock { requested: 2048, limit: 1024 };
        let s = format!("{e}");
        assert!(s.contains("2048"));
        assert!(s.contains("1024"));
    }

    #[test]
    fn test_error_display_grid() {
        let e = LaunchError::ExceedsMaxGridDim { axis: "y", requested: 70_000, limit: 65_535 };
        assert!(format!("{e}").contains("y"));
    }

    #[test]
    fn test_error_display_shared_mem() {
        let e = LaunchError::ExceedsSharedMemory { requested: 200_000, limit: 164_000 };
        assert!(format!("{e}").contains("200000"));
    }

    #[test]
    fn test_error_display_zero_block() {
        let e = LaunchError::ZeroBlockDim { axis: "x" };
        assert!(format!("{e}").contains("x"));
    }

    #[test]
    fn test_error_display_zero_work() {
        let e = LaunchError::ZeroWorkSize;
        assert!(format!("{e}").contains("zero"));
    }

    #[test]
    fn test_error_display_warp_aligned() {
        let e = LaunchError::NotWarpAligned { block_dim_x: 100, warp_size: 32 };
        assert!(format!("{e}").contains("100"));
    }

    // -- helpers ----------------------------------------------------

    #[test]
    fn test_next_pow2_up() {
        assert_eq!(next_pow2_up(0), 1);
        assert_eq!(next_pow2_up(1), 1);
        assert_eq!(next_pow2_up(2), 2);
        assert_eq!(next_pow2_up(3), 4);
        assert_eq!(next_pow2_up(255), 256);
        assert_eq!(next_pow2_up(256), 256);
        assert_eq!(next_pow2_up(257), 512);
    }

    #[test]
    fn test_next_pow2_down() {
        assert_eq!(next_pow2_down(0), 1);
        assert_eq!(next_pow2_down(1), 1);
        assert_eq!(next_pow2_down(2), 2);
        assert_eq!(next_pow2_down(3), 2);
        assert_eq!(next_pow2_down(255), 128);
        assert_eq!(next_pow2_down(256), 256);
        assert_eq!(next_pow2_down(1023), 512);
    }

    #[test]
    fn test_clamp_to_warp_below() {
        assert_eq!(clamp_to_warp(1), WARP_SIZE);
        assert_eq!(clamp_to_warp(0), WARP_SIZE);
    }

    #[test]
    fn test_clamp_to_warp_above() {
        assert_eq!(clamp_to_warp(2048), 1024);
    }

    #[test]
    fn test_clamp_to_warp_rounds_down() {
        assert_eq!(clamp_to_warp(100), 96); // 3*32
    }

    // -- integration-style ------------------------------------------

    #[test]
    fn test_end_to_end_validate_computed_config() {
        let cfg = compute_launch_config(50_000, 256);
        assert!(validate_launch_config(&cfg, &DeviceLimits::a100()).is_ok());
    }

    #[test]
    fn test_end_to_end_2d_validate() {
        let cfg = compute_2d_launch(1080, 1920, 16, 16);
        assert!(validate_launch_config(&cfg, &DeviceLimits::rtx_5070ti()).is_ok());
    }

    #[test]
    fn test_end_to_end_3d_validate() {
        let cfg = compute_3d_launch(64, 64, 64, (8, 8, 8));
        assert!(validate_launch_config(&cfg, &DeviceLimits::generic_sm89()).is_ok());
    }

    #[test]
    fn test_optimal_then_launch_then_validate() {
        let bs = optimal_block_size(100_000, 0);
        let cfg = compute_launch_config(100_000, bs);
        assert!(validate_launch_config(&cfg, &DeviceLimits::a100()).is_ok());
    }

    #[test]
    fn test_work_distribution_matches_config() {
        let n = 9999;
        let bs = 256u32;
        let cfg = compute_launch_config(n, bs);
        let wd = work_distribution(n, bs);
        assert_eq!(cfg.grid_dim.0, wd.blocks);
        assert_eq!(cfg.block_dim.0, wd.threads_per_block);
    }

    // -- edge cases -------------------------------------------------

    #[test]
    fn test_2d_launch_large_grid() {
        let cfg = compute_2d_launch(65_535, 65_535, 1, 1);
        assert_eq!(cfg.grid_dim.0, 65_535);
        assert_eq!(cfg.grid_dim.1, 65_535);
    }

    #[test]
    fn test_3d_launch_one_each() {
        let cfg = compute_3d_launch(1, 1, 1, (1, 1, 1));
        assert_eq!(cfg.grid_dim, (1, 1, 1));
        assert_eq!(cfg.block_dim, (1, 1, 1));
    }

    #[test]
    fn test_launch_config_with_custom_shared_mem() {
        let mut cfg = compute_launch_config(1024, 256);
        cfg.shared_mem_bytes = 4096;
        assert_eq!(cfg.shared_mem_bytes, 4096);
        assert!(validate_launch_config(&cfg, &DeviceLimits::a100()).is_ok());
    }

    #[test]
    fn test_launch_config_with_stream() {
        let mut cfg = compute_launch_config(512, 128);
        cfg.stream_id = Some(7);
        assert_eq!(cfg.stream_id, Some(7));
    }

    #[test]
    fn test_total_threads_2d() {
        let cfg = compute_2d_launch(4, 8, 2, 2);
        // grid=(4,2,1), block=(2,2,1) → 4*2*1 * 2*2*1 = 32
        assert_eq!(cfg.total_threads(), 32);
    }

    #[test]
    fn test_total_threads_3d() {
        let cfg = compute_3d_launch(4, 6, 8, (2, 3, 4));
        // grid=(2,2,2), block=(2,3,4) → 8 * 24 = 192
        assert_eq!(cfg.total_threads(), 192);
    }

    #[test]
    fn test_validate_grid_x_boundary() {
        let cfg = LaunchConfig {
            grid_dim: (MAX_GRID_DIM_X, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
            stream_id: None,
        };
        assert!(validate_launch_config(&cfg, &DeviceLimits::a100()).is_ok());
    }

    #[test]
    fn test_validate_all_presets_accept_typical() {
        let cfg = compute_launch_config(10_000, 256);
        for limits in [
            DeviceLimits::a100(),
            DeviceLimits::rtx_5070ti(),
            DeviceLimits::generic_sm89(),
            DeviceLimits::conservative(),
        ] {
            assert!(validate_launch_config(&cfg, &limits).is_ok(), "failed for {limits:?}");
        }
    }

    #[test]
    fn test_launch_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(LaunchError::ZeroWorkSize);
        assert!(!e.to_string().is_empty());
    }
}
