//! Launch configuration and occupancy-based tuning.

use crate::{
    DEFAULT_BLOCK_SIZE, DEFAULT_SHARED_MEM_BYTES, MAX_GRID_DIM_X, MAX_GRID_DIM_YZ,
    MAX_THREADS_PER_BLOCK, WARP_SIZE, clamp_block_size, grid_blocks_1d,
};

// ── OccupancyHint ──────────────────────────────────────────────────────────

/// Hints for occupancy-based launch configuration.
///
/// CUDA occupancy depends on registers-per-thread and shared-memory-per-block.
/// This struct carries those hints so [`LaunchConfig`] can pick a block size
/// that maximises SM utilisation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OccupancyHint {
    /// Registers used per thread (0 = unknown / let driver decide).
    pub regs_per_thread: u32,
    /// Dynamic shared memory bytes per block.
    pub shared_mem_bytes: u32,
    /// Target occupancy ratio (`0.0..=1.0`). `None` = maximise.
    pub target_occupancy: Option<f64>,
}

impl OccupancyHint {
    /// Create a hint with only shared memory specified.
    #[must_use]
    pub const fn with_shared_mem(shared_mem_bytes: u32) -> Self {
        Self { regs_per_thread: 0, shared_mem_bytes, target_occupancy: None }
    }

    /// Estimate a good block size based on register and shared-memory pressure.
    ///
    /// This is a heuristic, not a replacement for
    /// `cudaOccupancyMaxPotentialBlockSize`.
    #[must_use]
    pub fn suggested_block_size(&self) -> u32 {
        let mut block = DEFAULT_BLOCK_SIZE;

        // High register usage → fewer concurrent warps → smaller block
        if self.regs_per_thread > 64 {
            block = 128;
        } else if self.regs_per_thread > 32 {
            block = 192;
        }

        // Excessive shared memory → reduce block to stay within per-SM limits
        if self.shared_mem_bytes > DEFAULT_SHARED_MEM_BYTES / 2 {
            block = block.min(128);
        }

        // Target-occupancy trim: low targets allow larger blocks for caching
        if let Some(t) = self.target_occupancy
            && t < 0.5
        {
            block = block.max(512);
        }

        clamp_block_size(block)
    }
}

// ── LaunchConfig ───────────────────────────────────────────────────────────

/// Complete CUDA kernel launch configuration: grid dims, block dims, and
/// dynamic shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Grid dimensions (`blocks_x`, `blocks_y`, `blocks_z`).
    pub grid: (u32, u32, u32),
    /// Block dimensions (`threads_x`, `threads_y`, `threads_z`).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    // ── 1-D constructors ───────────────────────────────────────────────

    /// Build a 1-D launch config for `n` elements.
    #[must_use]
    pub fn for_elements(n: u64) -> Self {
        Self::for_elements_with_block(n, DEFAULT_BLOCK_SIZE)
    }

    /// Build a 1-D launch config with a custom block size.
    #[must_use]
    pub fn for_elements_with_block(n: u64, block_size: u32) -> Self {
        let bs = clamp_block_size(block_size);
        Self { grid: (grid_blocks_1d(n, bs), 1, 1), block: (bs, 1, 1), shared_mem_bytes: 0 }
    }

    /// Build a 1-D config guided by occupancy hints.
    #[must_use]
    pub fn for_elements_with_hint(n: u64, hint: &OccupancyHint) -> Self {
        let bs = hint.suggested_block_size();
        Self {
            grid: (grid_blocks_1d(n, bs), 1, 1),
            block: (bs, 1, 1),
            shared_mem_bytes: hint.shared_mem_bytes,
        }
    }

    // ── 2-D constructors ───────────────────────────────────────────────

    /// Build a 2-D launch config for a matrix of (`rows`, `cols`).
    ///
    /// `grid.x` covers columns, `grid.y` covers rows — matching typical
    /// CUDA layout.
    #[must_use]
    pub fn for_matrix(rows: u32, cols: u32) -> Self {
        Self::for_matrix_with_tile(rows, cols, 16, 16)
    }

    /// Build a 2-D launch config with custom tile dimensions.
    #[must_use]
    pub fn for_matrix_with_tile(rows: u32, cols: u32, tile_x: u32, tile_y: u32) -> Self {
        let tx = tile_x.clamp(1, MAX_THREADS_PER_BLOCK);
        let ty = tile_y.clamp(1, MAX_THREADS_PER_BLOCK / tx);

        let gx = cols.div_ceil(tx).min(MAX_GRID_DIM_X);
        let gy = rows.div_ceil(ty).min(MAX_GRID_DIM_YZ);

        Self { grid: (gx, gy, 1), block: (tx, ty, 1), shared_mem_bytes: 0 }
    }

    // ── 3-D constructor ────────────────────────────────────────────────

    /// Build a 3-D launch config for a volume of (`x`, `y`, `z`) elements.
    #[must_use]
    pub fn for_volume(x: u32, y: u32, z: u32, threads_per_dim: u32) -> Self {
        let t = threads_per_dim.clamp(1, 10); // keep total ≤ 1000
        let gx = x.div_ceil(t).min(MAX_GRID_DIM_X);
        let gy = y.div_ceil(t).min(MAX_GRID_DIM_YZ);
        let gz = z.div_ceil(t).min(MAX_GRID_DIM_YZ);
        Self { grid: (gx, gy, gz), block: (t, t, t), shared_mem_bytes: 0 }
    }

    // ── Row-per-block (softmax-style) ──────────────────────────────────

    /// One block per row, threads cover columns (softmax / layer-norm pattern).
    #[must_use]
    pub fn row_per_block(n_rows: u32, n_cols: u32) -> Self {
        let threads = n_cols.min(MAX_THREADS_PER_BLOCK);
        let threads = clamp_block_size(threads);
        Self {
            grid: (n_rows.min(MAX_GRID_DIM_X), 1, 1),
            block: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    // ── Shared memory builder ──────────────────────────────────────────

    /// Attach dynamic shared memory to this config.
    #[must_use]
    pub const fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Total thread count =
    /// `grid.x * grid.y * grid.z * block.x * block.y * block.z`.
    #[must_use]
    pub const fn total_threads(&self) -> u64 {
        let grid_total = self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64;
        let block_total = self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64;
        grid_total * block_total
    }

    /// Number of warps per block.
    #[must_use]
    pub const fn warps_per_block(&self) -> u32 {
        let threads = self.block.0 * self.block.1 * self.block.2;
        threads.div_ceil(WARP_SIZE)
    }

    /// Whether this config is valid for CUDA launch.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        let threads = self.block.0 * self.block.1 * self.block.2;
        threads > 0
            && threads <= MAX_THREADS_PER_BLOCK
            && self.grid.0 > 0
            && self.grid.1 > 0
            && self.grid.2 > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LaunchConfig 1-D ───────────────────────────────────────────────

    #[test]
    fn config_1d_small() {
        let c = LaunchConfig::for_elements(100);
        assert_eq!(c.grid, (1, 1, 1));
        assert_eq!(c.block, (DEFAULT_BLOCK_SIZE, 1, 1));
        assert!(c.is_valid());
    }

    #[test]
    fn config_1d_exact() {
        let c = LaunchConfig::for_elements(256);
        assert_eq!(c.grid, (1, 1, 1));
    }

    #[test]
    fn config_1d_large() {
        let c = LaunchConfig::for_elements(1_000_000);
        assert_eq!(c.grid.0, 3907); // ceil(1M / 256)
        assert!(c.is_valid());
    }

    #[test]
    fn config_1d_custom_block() {
        let c = LaunchConfig::for_elements_with_block(1024, 512);
        assert_eq!(c.block.0, 512);
        assert_eq!(c.grid.0, 2);
    }

    #[test]
    fn config_1d_with_hint() {
        let hint =
            OccupancyHint { regs_per_thread: 80, shared_mem_bytes: 1024, ..Default::default() };
        let c = LaunchConfig::for_elements_with_hint(512, &hint);
        assert_eq!(c.block.0, 128); // high regs → small block
        assert_eq!(c.shared_mem_bytes, 1024);
        assert!(c.is_valid());
    }

    // ── LaunchConfig 2-D ───────────────────────────────────────────────

    #[test]
    fn config_2d_square() {
        let c = LaunchConfig::for_matrix(64, 64);
        assert_eq!(c.grid, (4, 4, 1));
        assert_eq!(c.block, (16, 16, 1));
    }

    #[test]
    fn config_2d_non_divisible() {
        let c = LaunchConfig::for_matrix(17, 33);
        assert_eq!(c.grid, (3, 2, 1)); // ceil(33/16), ceil(17/16)
        assert_eq!(c.block, (16, 16, 1));
    }

    #[test]
    fn config_2d_custom_tile() {
        let c = LaunchConfig::for_matrix_with_tile(100, 200, 32, 8);
        assert_eq!(c.block, (32, 8, 1));
        assert_eq!(c.grid.0, 7); // ceil(200/32)
        assert_eq!(c.grid.1, 13); // ceil(100/8)
    }

    #[test]
    fn config_2d_single_element() {
        let c = LaunchConfig::for_matrix(1, 1);
        assert_eq!(c.grid, (1, 1, 1));
        assert!(c.is_valid());
    }

    // ── LaunchConfig 3-D ───────────────────────────────────────────────

    #[test]
    fn config_3d_basic() {
        let c = LaunchConfig::for_volume(32, 32, 32, 8);
        assert_eq!(c.block, (8, 8, 8));
        assert_eq!(c.grid, (4, 4, 4));
        assert!(c.is_valid());
    }

    // ── Row-per-block ──────────────────────────────────────────────────

    #[test]
    fn row_per_block_basic() {
        let c = LaunchConfig::row_per_block(8, 128);
        assert_eq!(c.grid.0, 8);
        assert_eq!(c.block.0, 128);
    }

    #[test]
    fn row_per_block_wide() {
        let c = LaunchConfig::row_per_block(4, 4096);
        assert_eq!(c.block.0, MAX_THREADS_PER_BLOCK);
    }

    // ── Shared memory ──────────────────────────────────────────────────

    #[test]
    fn shared_mem_builder() {
        let c = LaunchConfig::for_elements(512).with_shared_mem(2048);
        assert_eq!(c.shared_mem_bytes, 2048);
    }

    // ── Accessors ──────────────────────────────────────────────────────

    #[test]
    fn total_threads_1d() {
        let c = LaunchConfig::for_elements(512);
        assert_eq!(c.total_threads(), 512);
    }

    #[test]
    fn total_threads_2d() {
        let c = LaunchConfig::for_matrix(32, 32);
        assert_eq!(c.total_threads(), 2 * 2 * 16 * 16);
    }

    #[test]
    fn warps_per_block_256() {
        let c = LaunchConfig::for_elements(256);
        assert_eq!(c.warps_per_block(), 8); // 256/32
    }

    // ── OccupancyHint ──────────────────────────────────────────────────

    #[test]
    fn hint_default_block() {
        let h = OccupancyHint::default();
        assert_eq!(h.suggested_block_size(), DEFAULT_BLOCK_SIZE);
    }

    #[test]
    fn hint_high_regs() {
        let h = OccupancyHint { regs_per_thread: 100, ..Default::default() };
        assert_eq!(h.suggested_block_size(), 128);
    }

    #[test]
    fn hint_medium_regs() {
        let h = OccupancyHint { regs_per_thread: 48, ..Default::default() };
        assert_eq!(h.suggested_block_size(), 192);
    }

    #[test]
    fn hint_heavy_shared_mem() {
        let h =
            OccupancyHint { regs_per_thread: 0, shared_mem_bytes: 30_000, target_occupancy: None };
        assert_eq!(h.suggested_block_size(), 128);
    }

    #[test]
    fn hint_low_target_occupancy() {
        let h =
            OccupancyHint { regs_per_thread: 0, shared_mem_bytes: 0, target_occupancy: Some(0.3) };
        assert_eq!(h.suggested_block_size(), 512);
    }

    #[test]
    fn hint_with_shared_mem_constructor() {
        let h = OccupancyHint::with_shared_mem(4096);
        assert_eq!(h.shared_mem_bytes, 4096);
        assert_eq!(h.regs_per_thread, 0);
    }

    // ── Validity ───────────────────────────────────────────────────────

    #[test]
    fn valid_config() {
        assert!(LaunchConfig::for_elements(1).is_valid());
        assert!(LaunchConfig::for_matrix(1, 1).is_valid());
    }
}
