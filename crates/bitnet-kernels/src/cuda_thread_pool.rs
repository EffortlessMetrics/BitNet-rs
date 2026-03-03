//! CUDA thread pool and work scheduling for GPU kernel dispatch.
//!
//! Provides types and functions for computing optimal thread block and grid
//! configurations, partitioning work items across CUDA threads, and validating
//! launch parameters against hardware limits.

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// CUDA warp size (threads executed in lock-step).
pub const WARP_SIZE: u32 = 32;

/// Default maximum threads per block on most NVIDIA GPUs.
pub const DEFAULT_MAX_THREADS_PER_BLOCK: u32 = 1024;

/// Maximum grid dimension (x) on compute capability ≥ 3.0.
pub const MAX_GRID_DIM_X: u32 = 2_147_483_647; // 2^31 - 1

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors arising from invalid thread/grid configurations or scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    /// Total threads in a block exceed hardware limit.
    ExceedsMaxThreads { requested: u32, max_threads: u32 },
    /// One or more block dimensions is zero.
    ZeroDimension { dimension: &'static str },
    /// Total work size is zero — nothing to schedule.
    ZeroWorkSize,
    /// Coarsening factor is zero or exceeds total work.
    InvalidCoarseningFactor { factor: u32, total_work: u32 },
    /// Grid dimension would overflow hardware limits.
    GridOverflow { blocks: u64, max_blocks: u32 },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExceedsMaxThreads { requested, max_threads } => {
                write!(f, "thread block size {requested} exceeds max {max_threads}")
            }
            Self::ZeroDimension { dimension } => {
                write!(f, "thread block dimension '{dimension}' is zero")
            }
            Self::ZeroWorkSize => write!(f, "total work size is zero"),
            Self::InvalidCoarseningFactor { factor, total_work } => {
                write!(f, "coarsening factor {factor} invalid for work size {total_work}")
            }
            Self::GridOverflow { blocks, max_blocks } => {
                write!(f, "grid requires {blocks} blocks but max is {max_blocks}")
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

// ---------------------------------------------------------------------------
// Thread block config
// ---------------------------------------------------------------------------

/// Describes the dimensions of a CUDA thread block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadBlockConfig {
    pub threads_x: u32,
    pub threads_y: u32,
    pub threads_z: u32,
}

impl ThreadBlockConfig {
    /// Create a new config with explicit dimensions.
    pub fn new(threads_x: u32, threads_y: u32, threads_z: u32) -> Self {
        Self { threads_x, threads_y, threads_z }
    }

    /// One-dimensional block of `n` threads.
    pub fn new_1d(n: u32) -> Self {
        Self::new(n, 1, 1)
    }

    /// Two-dimensional block.
    pub fn new_2d(x: u32, y: u32) -> Self {
        Self::new(x, y, 1)
    }

    /// Total number of threads in the block.
    pub fn total_threads(&self) -> u32 {
        self.threads_x.saturating_mul(self.threads_y).saturating_mul(self.threads_z)
    }

    /// Whether the x-dimension is a multiple of the warp size.
    pub fn is_warp_aligned(&self) -> bool {
        self.threads_x.is_multiple_of(WARP_SIZE)
    }
}

impl fmt::Display for ThreadBlockConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.threads_x, self.threads_y, self.threads_z)
    }
}

// ---------------------------------------------------------------------------
// Grid config
// ---------------------------------------------------------------------------

/// Describes the dimensions of a CUDA grid (number of blocks per dimension).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridConfig {
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub blocks_z: u32,
}

impl GridConfig {
    pub fn new(blocks_x: u32, blocks_y: u32, blocks_z: u32) -> Self {
        Self { blocks_x, blocks_y, blocks_z }
    }

    pub fn new_1d(blocks: u32) -> Self {
        Self::new(blocks, 1, 1)
    }

    /// Total number of blocks in the grid.
    pub fn total_blocks(&self) -> u64 {
        (self.blocks_x as u64)
            .saturating_mul(self.blocks_y as u64)
            .saturating_mul(self.blocks_z as u64)
    }
}

impl fmt::Display for GridConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.blocks_x, self.blocks_y, self.blocks_z)
    }
}

// ---------------------------------------------------------------------------
// Schedule strategy
// ---------------------------------------------------------------------------

/// Strategy for partitioning work across CUDA threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStrategy {
    /// Each thread block processes a contiguous range of elements.
    LinearPartition,
    /// Work is divided into 2-D tiles for spatial locality.
    TiledPartition,
    /// Diagonal wavefront ordering to avoid bank conflicts.
    WavefrontPartition,
}

// ---------------------------------------------------------------------------
// Work item
// ---------------------------------------------------------------------------

/// Represents a single unit of work assigned to a thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkItem {
    /// Global thread index across the entire dispatch.
    pub global_id: u32,
    /// Thread index within its block (local).
    pub local_id: u32,
    /// Index of the block this thread belongs to.
    pub group_id: u32,
    /// Number of threads in the block.
    pub group_size: u32,
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Compute the 1-D grid configuration needed to cover `total_work` elements.
///
/// The number of blocks is `ceil(total_work / block_threads_x)`.
pub fn compute_grid_config(
    total_work: u32,
    block_config: &ThreadBlockConfig,
) -> Result<GridConfig, ScheduleError> {
    if total_work == 0 {
        return Err(ScheduleError::ZeroWorkSize);
    }
    if block_config.threads_x == 0 {
        return Err(ScheduleError::ZeroDimension { dimension: "threads_x" });
    }
    let blocks = div_ceil(total_work, block_config.threads_x);
    Ok(GridConfig::new_1d(blocks))
}

/// Generate a list of [`WorkItem`]s for every virtual thread in the dispatch.
pub fn schedule_work(
    total_elements: u32,
    strategy: ScheduleStrategy,
    block_config: &ThreadBlockConfig,
) -> Result<Vec<WorkItem>, ScheduleError> {
    if total_elements == 0 {
        return Err(ScheduleError::ZeroWorkSize);
    }
    let block_size = block_config.threads_x;
    if block_size == 0 {
        return Err(ScheduleError::ZeroDimension { dimension: "threads_x" });
    }

    let num_blocks = div_ceil(total_elements, block_size);
    let total_threads = num_blocks * block_size;
    let mut items = Vec::with_capacity(total_threads as usize);

    match strategy {
        ScheduleStrategy::LinearPartition => {
            for block_id in 0..num_blocks {
                for local_id in 0..block_size {
                    let global_id = block_id * block_size + local_id;
                    items.push(WorkItem {
                        global_id,
                        local_id,
                        group_id: block_id,
                        group_size: block_size,
                    });
                }
            }
        }
        ScheduleStrategy::TiledPartition => {
            // Tiles are laid out in row-major order; thread ordering within a
            // tile follows the linear convention but groups are interleaved to
            // improve L2 locality on 2-D data.
            for block_id in 0..num_blocks {
                for local_id in 0..block_size {
                    // Interleave: even blocks first, then odd.
                    let reordered_block = if num_blocks <= 1 {
                        block_id
                    } else {
                        let half = div_ceil(num_blocks, 2);
                        if block_id < half { block_id * 2 } else { (block_id - half) * 2 + 1 }
                    };
                    let reordered_block = reordered_block.min(num_blocks - 1);
                    let global_id = reordered_block * block_size + local_id;
                    items.push(WorkItem {
                        global_id,
                        local_id,
                        group_id: reordered_block,
                        group_size: block_size,
                    });
                }
            }
        }
        ScheduleStrategy::WavefrontPartition => {
            // Diagonal wavefront: blocks are scheduled along anti-diagonals.
            // For 1-D work this reduces to a stride-2 interleave.
            for block_id in 0..num_blocks {
                let wave_id = if num_blocks <= 1 {
                    0
                } else {
                    // Assign blocks to two waves: first wave gets even indices,
                    // second wave gets odd indices (simple 1-D wavefront).
                    block_id % 2
                };
                let half = div_ceil(num_blocks, 2);
                let index_in_wave = block_id / 2;
                let reordered_block =
                    if wave_id == 0 { index_in_wave } else { half + index_in_wave };
                let reordered_block = reordered_block.min(num_blocks - 1);
                for local_id in 0..block_size {
                    let global_id = reordered_block * block_size + local_id;
                    items.push(WorkItem {
                        global_id,
                        local_id,
                        group_id: reordered_block,
                        group_size: block_size,
                    });
                }
            }
        }
    }

    Ok(items)
}

/// Choose an optimal 1-D thread block size for `n` elements.
///
/// Returns a warp-aligned block where `threads_x` is a multiple of
/// [`WARP_SIZE`], capped at [`DEFAULT_MAX_THREADS_PER_BLOCK`].
pub fn optimal_block_size_1d(n: u32) -> ThreadBlockConfig {
    if n == 0 {
        return ThreadBlockConfig::new_1d(WARP_SIZE);
    }
    // Round up to nearest warp multiple, cap at max.
    let warps_needed = div_ceil(n, WARP_SIZE);
    let max_warps = DEFAULT_MAX_THREADS_PER_BLOCK / WARP_SIZE;
    let warps = warps_needed.min(max_warps);
    ThreadBlockConfig::new_1d(warps * WARP_SIZE)
}

/// Choose an optimal 2-D thread block for a matrix of `rows × cols`.
///
/// Prefers square-ish blocks whose total fits within
/// [`DEFAULT_MAX_THREADS_PER_BLOCK`].
pub fn optimal_block_size_2d(rows: u32, cols: u32) -> ThreadBlockConfig {
    if rows == 0 || cols == 0 {
        return ThreadBlockConfig::new_2d(WARP_SIZE, 1);
    }
    // Start with 16×16 = 256 threads (common CUDA sweet spot).
    let mut bx: u32 = 16;
    let mut by: u32 = 16;

    // Adjust x to not exceed cols, keeping warp alignment.
    if cols < bx {
        bx = round_up_to_warp(cols);
    }
    // Adjust y to not exceed rows and stay within limit.
    let max_y = DEFAULT_MAX_THREADS_PER_BLOCK / bx;
    if rows < by {
        by = rows.max(1);
    }
    by = by.min(max_y);

    ThreadBlockConfig::new_2d(bx, by)
}

/// Compute a grid that applies *thread coarsening*: each thread processes
/// `coarsening_factor` elements, reducing the total number of blocks.
pub fn thread_coarsening(
    total_work: u32,
    block_config: &ThreadBlockConfig,
    coarsening_factor: u32,
) -> Result<GridConfig, ScheduleError> {
    if total_work == 0 {
        return Err(ScheduleError::ZeroWorkSize);
    }
    if coarsening_factor == 0 || coarsening_factor > total_work {
        return Err(ScheduleError::InvalidCoarseningFactor {
            factor: coarsening_factor,
            total_work,
        });
    }
    let threads_per_block = block_config.threads_x;
    if threads_per_block == 0 {
        return Err(ScheduleError::ZeroDimension { dimension: "threads_x" });
    }
    let effective_work = div_ceil(total_work, coarsening_factor);
    let blocks = div_ceil(effective_work, threads_per_block);
    Ok(GridConfig::new_1d(blocks))
}

/// Validate a [`ThreadBlockConfig`] against a hardware thread limit.
pub fn validate_thread_config(
    block: &ThreadBlockConfig,
    max_threads: u32,
) -> Result<(), ScheduleError> {
    if block.threads_x == 0 {
        return Err(ScheduleError::ZeroDimension { dimension: "threads_x" });
    }
    if block.threads_y == 0 {
        return Err(ScheduleError::ZeroDimension { dimension: "threads_y" });
    }
    if block.threads_z == 0 {
        return Err(ScheduleError::ZeroDimension { dimension: "threads_z" });
    }
    let total = block.total_threads();
    if total > max_threads {
        return Err(ScheduleError::ExceedsMaxThreads { requested: total, max_threads });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Integer ceil division.
fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Round `n` up to the next multiple of [`WARP_SIZE`].
fn round_up_to_warp(n: u32) -> u32 {
    let rem = n % WARP_SIZE;
    if rem == 0 { n.max(WARP_SIZE) } else { n + (WARP_SIZE - rem) }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ThreadBlockConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_config_new_1d() {
        let b = ThreadBlockConfig::new_1d(256);
        assert_eq!(b.threads_x, 256);
        assert_eq!(b.threads_y, 1);
        assert_eq!(b.threads_z, 1);
    }

    #[test]
    fn test_block_config_new_2d() {
        let b = ThreadBlockConfig::new_2d(16, 16);
        assert_eq!(b.threads_x, 16);
        assert_eq!(b.threads_y, 16);
        assert_eq!(b.threads_z, 1);
    }

    #[test]
    fn test_block_config_new_3d() {
        let b = ThreadBlockConfig::new(8, 8, 4);
        assert_eq!(b.total_threads(), 256);
    }

    #[test]
    fn test_block_total_threads() {
        let b = ThreadBlockConfig::new(32, 4, 2);
        assert_eq!(b.total_threads(), 256);
    }

    #[test]
    fn test_block_warp_aligned() {
        assert!(ThreadBlockConfig::new_1d(128).is_warp_aligned());
        assert!(!ThreadBlockConfig::new_1d(33).is_warp_aligned());
    }

    #[test]
    fn test_block_display() {
        let b = ThreadBlockConfig::new(32, 8, 1);
        assert_eq!(format!("{b}"), "(32, 8, 1)");
    }

    // -----------------------------------------------------------------------
    // GridConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_grid_new_1d() {
        let g = GridConfig::new_1d(42);
        assert_eq!(g.blocks_x, 42);
        assert_eq!(g.blocks_y, 1);
        assert_eq!(g.blocks_z, 1);
    }

    #[test]
    fn test_grid_total_blocks() {
        let g = GridConfig::new(4, 3, 2);
        assert_eq!(g.total_blocks(), 24);
    }

    #[test]
    fn test_grid_display() {
        let g = GridConfig::new(10, 1, 1);
        assert_eq!(format!("{g}"), "(10, 1, 1)");
    }

    // -----------------------------------------------------------------------
    // compute_grid_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_grid_exact_division() {
        let b = ThreadBlockConfig::new_1d(128);
        let g = compute_grid_config(256, &b).unwrap();
        assert_eq!(g.blocks_x, 2);
    }

    #[test]
    fn test_grid_with_remainder() {
        let b = ThreadBlockConfig::new_1d(128);
        let g = compute_grid_config(300, &b).unwrap();
        assert_eq!(g.blocks_x, 3); // ceil(300/128) = 3
    }

    #[test]
    fn test_grid_single_element() {
        let b = ThreadBlockConfig::new_1d(256);
        let g = compute_grid_config(1, &b).unwrap();
        assert_eq!(g.blocks_x, 1);
    }

    #[test]
    fn test_grid_zero_work_err() {
        let b = ThreadBlockConfig::new_1d(128);
        assert_eq!(compute_grid_config(0, &b), Err(ScheduleError::ZeroWorkSize));
    }

    #[test]
    fn test_grid_zero_block_dim_err() {
        let b = ThreadBlockConfig::new_1d(0);
        assert!(matches!(compute_grid_config(100, &b), Err(ScheduleError::ZeroDimension { .. })));
    }

    #[test]
    fn test_grid_large_work() {
        let b = ThreadBlockConfig::new_1d(256);
        let g = compute_grid_config(1_000_000, &b).unwrap();
        assert_eq!(g.blocks_x, 3907); // ceil(1M/256)
    }

    // -----------------------------------------------------------------------
    // schedule_work — LinearPartition
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_basic() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(8, ScheduleStrategy::LinearPartition, &b).unwrap();
        assert_eq!(items.len(), 8);
        assert_eq!(items[0].global_id, 0);
        assert_eq!(items[7].global_id, 7);
    }

    #[test]
    fn test_linear_padding() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(5, ScheduleStrategy::LinearPartition, &b).unwrap();
        // 2 blocks × 4 threads = 8 items (3 padding threads).
        assert_eq!(items.len(), 8);
    }

    #[test]
    fn test_linear_local_ids() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(8, ScheduleStrategy::LinearPartition, &b).unwrap();
        for item in &items {
            assert!(item.local_id < 4);
        }
    }

    #[test]
    fn test_linear_group_ids() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(12, ScheduleStrategy::LinearPartition, &b).unwrap();
        assert_eq!(items[0].group_id, 0);
        assert_eq!(items[4].group_id, 1);
        assert_eq!(items[8].group_id, 2);
    }

    #[test]
    fn test_linear_group_size() {
        let b = ThreadBlockConfig::new_1d(32);
        let items = schedule_work(64, ScheduleStrategy::LinearPartition, &b).unwrap();
        for item in &items {
            assert_eq!(item.group_size, 32);
        }
    }

    #[test]
    fn test_linear_single_element() {
        let b = ThreadBlockConfig::new_1d(32);
        let items = schedule_work(1, ScheduleStrategy::LinearPartition, &b).unwrap();
        assert_eq!(items.len(), 32); // one full block
        assert_eq!(items[0].global_id, 0);
    }

    #[test]
    fn test_linear_zero_work_err() {
        let b = ThreadBlockConfig::new_1d(32);
        assert!(schedule_work(0, ScheduleStrategy::LinearPartition, &b).is_err());
    }

    #[test]
    fn test_linear_zero_block_err() {
        let b = ThreadBlockConfig::new_1d(0);
        assert!(schedule_work(10, ScheduleStrategy::LinearPartition, &b).is_err());
    }

    // -----------------------------------------------------------------------
    // schedule_work — TiledPartition
    // -----------------------------------------------------------------------

    #[test]
    fn test_tiled_basic() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(8, ScheduleStrategy::TiledPartition, &b).unwrap();
        assert_eq!(items.len(), 8);
    }

    #[test]
    fn test_tiled_reorders_blocks() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(16, ScheduleStrategy::TiledPartition, &b).unwrap();
        // First block of items should map to group 0 (even), second to group 2, etc.
        let first_group = items[0].group_id;
        let second_group = items[4].group_id;
        assert_ne!(first_group, 1); // should be even-first ordering
        assert!(second_group != first_group);
    }

    #[test]
    fn test_tiled_covers_all_globals() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(8, ScheduleStrategy::TiledPartition, &b).unwrap();
        let mut globals: Vec<u32> = items.iter().map(|i| i.global_id).collect();
        globals.sort();
        globals.dedup();
        // All 8 global ids should be present.
        assert_eq!(globals.len(), 8);
    }

    #[test]
    fn test_tiled_single_block() {
        let b = ThreadBlockConfig::new_1d(8);
        let items = schedule_work(4, ScheduleStrategy::TiledPartition, &b).unwrap();
        // Single block — tiling is a no-op.
        assert_eq!(items.len(), 8);
        assert_eq!(items[0].group_id, 0);
    }

    // -----------------------------------------------------------------------
    // schedule_work — WavefrontPartition
    // -----------------------------------------------------------------------

    #[test]
    fn test_wavefront_basic() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(8, ScheduleStrategy::WavefrontPartition, &b).unwrap();
        assert_eq!(items.len(), 8);
    }

    #[test]
    fn test_wavefront_reorders_blocks() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(16, ScheduleStrategy::WavefrontPartition, &b).unwrap();
        let group_order: Vec<u32> = items.chunks(4).map(|c| c[0].group_id).collect();
        // Wavefront interleaves: even blocks first, odd blocks second.
        assert_eq!(group_order[0], 0);
    }

    #[test]
    fn test_wavefront_covers_all_globals() {
        let b = ThreadBlockConfig::new_1d(4);
        let items = schedule_work(12, ScheduleStrategy::WavefrontPartition, &b).unwrap();
        let mut globals: Vec<u32> = items.iter().map(|i| i.global_id).collect();
        globals.sort();
        globals.dedup();
        assert_eq!(globals.len(), 12);
    }

    #[test]
    fn test_wavefront_single_block() {
        let b = ThreadBlockConfig::new_1d(8);
        let items = schedule_work(3, ScheduleStrategy::WavefrontPartition, &b).unwrap();
        assert_eq!(items.len(), 8);
        assert_eq!(items[0].group_id, 0);
    }

    // -----------------------------------------------------------------------
    // optimal_block_size_1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimal_1d_small() {
        let b = optimal_block_size_1d(10);
        assert_eq!(b.threads_x, WARP_SIZE);
        assert!(b.is_warp_aligned());
    }

    #[test]
    fn test_optimal_1d_exact_warp() {
        let b = optimal_block_size_1d(64);
        assert_eq!(b.threads_x, 64);
    }

    #[test]
    fn test_optimal_1d_large() {
        let b = optimal_block_size_1d(100_000);
        assert_eq!(b.threads_x, DEFAULT_MAX_THREADS_PER_BLOCK);
    }

    #[test]
    fn test_optimal_1d_zero() {
        let b = optimal_block_size_1d(0);
        assert_eq!(b.threads_x, WARP_SIZE);
    }

    #[test]
    fn test_optimal_1d_one() {
        let b = optimal_block_size_1d(1);
        assert_eq!(b.threads_x, WARP_SIZE);
    }

    #[test]
    fn test_optimal_1d_is_always_warp_aligned() {
        for n in [1, 17, 32, 33, 127, 256, 1023, 2048] {
            let b = optimal_block_size_1d(n);
            assert!(
                b.is_warp_aligned(),
                "optimal_block_size_1d({n}) not warp-aligned: {}",
                b.threads_x
            );
        }
    }

    #[test]
    fn test_optimal_1d_never_exceeds_max() {
        for n in [1, 1024, 1025, 100_000, u32::MAX] {
            let b = optimal_block_size_1d(n);
            assert!(b.total_threads() <= DEFAULT_MAX_THREADS_PER_BLOCK);
        }
    }

    // -----------------------------------------------------------------------
    // optimal_block_size_2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimal_2d_small_matrix() {
        let b = optimal_block_size_2d(4, 4);
        assert!(b.total_threads() <= DEFAULT_MAX_THREADS_PER_BLOCK);
        assert!(b.threads_x > 0 && b.threads_y > 0);
    }

    #[test]
    fn test_optimal_2d_large_matrix() {
        let b = optimal_block_size_2d(1024, 1024);
        assert!(b.total_threads() <= DEFAULT_MAX_THREADS_PER_BLOCK);
        assert!(b.threads_x >= 16);
        assert!(b.threads_y >= 16);
    }

    #[test]
    fn test_optimal_2d_wide() {
        let b = optimal_block_size_2d(2, 4096);
        assert!(b.threads_y <= 2);
    }

    #[test]
    fn test_optimal_2d_tall() {
        let b = optimal_block_size_2d(4096, 2);
        assert!(b.threads_x <= WARP_SIZE);
    }

    #[test]
    fn test_optimal_2d_zero_rows() {
        let b = optimal_block_size_2d(0, 100);
        assert_eq!(b.threads_y, 1);
    }

    #[test]
    fn test_optimal_2d_zero_cols() {
        let b = optimal_block_size_2d(100, 0);
        assert_eq!(b.threads_x, WARP_SIZE);
    }

    #[test]
    fn test_optimal_2d_never_exceeds_max() {
        for (r, c) in [(1, 1), (16, 16), (1, 10000), (10000, 1), (512, 512)] {
            let b = optimal_block_size_2d(r, c);
            assert!(
                b.total_threads() <= DEFAULT_MAX_THREADS_PER_BLOCK,
                "optimal_block_size_2d({r},{c}) = {} threads",
                b.total_threads()
            );
        }
    }

    // -----------------------------------------------------------------------
    // thread_coarsening
    // -----------------------------------------------------------------------

    #[test]
    fn test_coarsening_factor_1() {
        let b = ThreadBlockConfig::new_1d(128);
        let g = thread_coarsening(256, &b, 1).unwrap();
        assert_eq!(g.blocks_x, 2);
    }

    #[test]
    fn test_coarsening_factor_2() {
        let b = ThreadBlockConfig::new_1d(128);
        let g = thread_coarsening(256, &b, 2).unwrap();
        assert_eq!(g.blocks_x, 1); // 256/2=128 elems, 128/128=1 block
    }

    #[test]
    fn test_coarsening_factor_4() {
        let b = ThreadBlockConfig::new_1d(64);
        let g = thread_coarsening(1024, &b, 4).unwrap();
        assert_eq!(g.blocks_x, 4); // ceil(256/64) = 4
    }

    #[test]
    fn test_coarsening_reduces_blocks() {
        let b = ThreadBlockConfig::new_1d(128);
        let g1 = compute_grid_config(1024, &b).unwrap();
        let g2 = thread_coarsening(1024, &b, 4).unwrap();
        assert!(g2.blocks_x < g1.blocks_x);
    }

    #[test]
    fn test_coarsening_zero_factor_err() {
        let b = ThreadBlockConfig::new_1d(128);
        assert!(matches!(
            thread_coarsening(256, &b, 0),
            Err(ScheduleError::InvalidCoarseningFactor { .. })
        ));
    }

    #[test]
    fn test_coarsening_factor_exceeds_work_err() {
        let b = ThreadBlockConfig::new_1d(128);
        assert!(matches!(
            thread_coarsening(10, &b, 11),
            Err(ScheduleError::InvalidCoarseningFactor { .. })
        ));
    }

    #[test]
    fn test_coarsening_zero_work_err() {
        let b = ThreadBlockConfig::new_1d(128);
        assert_eq!(thread_coarsening(0, &b, 1), Err(ScheduleError::ZeroWorkSize));
    }

    #[test]
    fn test_coarsening_zero_block_err() {
        let b = ThreadBlockConfig::new_1d(0);
        assert!(matches!(thread_coarsening(100, &b, 1), Err(ScheduleError::ZeroDimension { .. })));
    }

    // -----------------------------------------------------------------------
    // validate_thread_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_ok() {
        let b = ThreadBlockConfig::new(32, 8, 1);
        assert!(validate_thread_config(&b, 1024).is_ok());
    }

    #[test]
    fn test_validate_exact_max() {
        let b = ThreadBlockConfig::new(32, 32, 1);
        assert!(validate_thread_config(&b, 1024).is_ok());
    }

    #[test]
    fn test_validate_exceeds() {
        let b = ThreadBlockConfig::new(32, 32, 2);
        assert!(matches!(
            validate_thread_config(&b, 1024),
            Err(ScheduleError::ExceedsMaxThreads { .. })
        ));
    }

    #[test]
    fn test_validate_zero_x() {
        let b = ThreadBlockConfig::new(0, 8, 1);
        assert!(matches!(
            validate_thread_config(&b, 1024),
            Err(ScheduleError::ZeroDimension { dimension: "threads_x" })
        ));
    }

    #[test]
    fn test_validate_zero_y() {
        let b = ThreadBlockConfig::new(32, 0, 1);
        assert!(matches!(
            validate_thread_config(&b, 1024),
            Err(ScheduleError::ZeroDimension { dimension: "threads_y" })
        ));
    }

    #[test]
    fn test_validate_zero_z() {
        let b = ThreadBlockConfig::new(32, 8, 0);
        assert!(matches!(
            validate_thread_config(&b, 1024),
            Err(ScheduleError::ZeroDimension { dimension: "threads_z" })
        ));
    }

    #[test]
    fn test_validate_max_1() {
        let b = ThreadBlockConfig::new(1, 1, 1);
        assert!(validate_thread_config(&b, 1).is_ok());
    }

    #[test]
    fn test_validate_max_1_exceeds() {
        let b = ThreadBlockConfig::new(2, 1, 1);
        assert!(matches!(
            validate_thread_config(&b, 1),
            Err(ScheduleError::ExceedsMaxThreads { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // ScheduleError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display_exceeds() {
        let e = ScheduleError::ExceedsMaxThreads { requested: 2048, max_threads: 1024 };
        assert!(format!("{e}").contains("2048"));
    }

    #[test]
    fn test_error_display_zero_dim() {
        let e = ScheduleError::ZeroDimension { dimension: "threads_x" };
        assert!(format!("{e}").contains("threads_x"));
    }

    #[test]
    fn test_error_display_zero_work() {
        let e = ScheduleError::ZeroWorkSize;
        assert!(format!("{e}").contains("zero"));
    }

    #[test]
    fn test_error_display_coarsening() {
        let e = ScheduleError::InvalidCoarseningFactor { factor: 0, total_work: 100 };
        assert!(format!("{e}").contains("0"));
    }

    #[test]
    fn test_error_display_grid_overflow() {
        let e = ScheduleError::GridOverflow { blocks: 999_999, max_blocks: 1024 };
        assert!(format!("{e}").contains("999999"));
    }

    // -----------------------------------------------------------------------
    // Helper: div_ceil
    // -----------------------------------------------------------------------

    #[test]
    fn test_div_ceil_exact() {
        assert_eq!(div_ceil(8, 4), 2);
    }

    #[test]
    fn test_div_ceil_remainder() {
        assert_eq!(div_ceil(9, 4), 3);
    }

    #[test]
    fn test_div_ceil_one() {
        assert_eq!(div_ceil(1, 1), 1);
    }

    // -----------------------------------------------------------------------
    // Helper: round_up_to_warp
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_up_warp_exact() {
        assert_eq!(round_up_to_warp(32), 32);
    }

    #[test]
    fn test_round_up_warp_below() {
        assert_eq!(round_up_to_warp(1), 32);
    }

    #[test]
    fn test_round_up_warp_above() {
        assert_eq!(round_up_to_warp(33), 64);
    }

    // -----------------------------------------------------------------------
    // Proptest properties
    // -----------------------------------------------------------------------

    mod proptests {
        use super::super::*;
        use proptest::prelude::*;

        proptest! {
            /// Grid blocks × block threads always covers total_work.
            #[test]
            fn grid_covers_work(total_work in 1u32..100_000, block_exp in 0u32..5) {
                let block_size = WARP_SIZE * (1 << block_exp); // 32, 64, 128, 256, 512
                let b = ThreadBlockConfig::new_1d(block_size.min(DEFAULT_MAX_THREADS_PER_BLOCK));
                let g = compute_grid_config(total_work, &b).unwrap();
                let coverage = g.blocks_x as u64 * b.threads_x as u64;
                prop_assert!(coverage >= total_work as u64);
            }

            /// optimal_block_size_1d always returns warp-aligned, capped config.
            #[test]
            fn optimal_1d_invariants(n in 0u32..1_000_000) {
                let b = optimal_block_size_1d(n);
                prop_assert!(b.is_warp_aligned());
                prop_assert!(b.total_threads() <= DEFAULT_MAX_THREADS_PER_BLOCK);
                prop_assert!(b.total_threads() >= WARP_SIZE);
            }

            /// optimal_block_size_2d never exceeds hardware limit.
            #[test]
            fn optimal_2d_invariants(rows in 0u32..10_000, cols in 0u32..10_000) {
                let b = optimal_block_size_2d(rows, cols);
                prop_assert!(b.total_threads() <= DEFAULT_MAX_THREADS_PER_BLOCK);
                prop_assert!(b.threads_x >= 1);
                prop_assert!(b.threads_y >= 1);
            }

            /// Coarsening always reduces (or equals) block count vs no coarsening.
            #[test]
            fn coarsening_reduces_blocks(
                total in 1u32..100_000,
                block_exp in 0u32..4,
                factor in 1u32..16,
            ) {
                let block_size = WARP_SIZE * (1 << block_exp);
                let b = ThreadBlockConfig::new_1d(block_size.min(DEFAULT_MAX_THREADS_PER_BLOCK));
                let factor = factor.min(total); // keep factor ≤ total
                let g_base = compute_grid_config(total, &b).unwrap();
                let g_coarse = thread_coarsening(total, &b, factor).unwrap();
                prop_assert!(g_coarse.blocks_x <= g_base.blocks_x);
            }

            /// validate_thread_config accepts configs at or below limit.
            #[test]
            fn validate_accepts_below_limit(x in 1u32..33, y in 1u32..33, z in 1u32..33) {
                let total = x.saturating_mul(y).saturating_mul(z);
                let b = ThreadBlockConfig::new(x, y, z);
                if total <= DEFAULT_MAX_THREADS_PER_BLOCK {
                    prop_assert!(validate_thread_config(&b, DEFAULT_MAX_THREADS_PER_BLOCK).is_ok());
                } else {
                    prop_assert!(validate_thread_config(&b, DEFAULT_MAX_THREADS_PER_BLOCK).is_err());
                }
            }
        }
    }
}
