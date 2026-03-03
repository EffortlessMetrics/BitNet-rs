//! Tile configuration for shared-memory CUDA transpose kernels.

/// Configuration for shared-memory tiled transpose.
///
/// A tile of size `tile_dim × tile_dim` is loaded into shared memory
/// and written out transposed, yielding coalesced global-memory
/// accesses in both the read and write phases.
///
/// The optional `block_rows` parameter allows each thread block to
/// process multiple rows per tile, reducing the total block count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    tile_dim: u32,
    block_rows: u32,
}

impl TileConfig {
    /// The recommended default: 32×32 tiles with 8 rows per block.
    pub const DEFAULT: Self = Self { tile_dim: 32, block_rows: 8 };

    /// Create a custom tile configuration.
    ///
    /// # Panics
    ///
    /// Panics if `tile_dim` is zero, `block_rows` is zero, or
    /// `tile_dim` is not divisible by `block_rows`.
    #[must_use]
    pub fn new(tile_dim: u32, block_rows: u32) -> Self {
        assert!(tile_dim > 0, "tile_dim must be > 0");
        assert!(block_rows > 0, "block_rows must be > 0");
        assert!(
            tile_dim.is_multiple_of(block_rows),
            "tile_dim ({tile_dim}) must be divisible by block_rows ({block_rows})"
        );
        Self { tile_dim, block_rows }
    }

    /// Tile dimension (width and height of the shared-memory tile).
    #[must_use]
    pub const fn tile_dim(&self) -> u32 {
        self.tile_dim
    }

    /// Number of rows each thread block processes per tile.
    #[must_use]
    pub const fn block_rows(&self) -> u32 {
        self.block_rows
    }

    /// Number of threads per block (`tile_dim × block_rows`).
    #[must_use]
    pub const fn threads_per_block(&self) -> u32 {
        self.tile_dim * self.block_rows
    }

    /// Grid dimensions required to cover a `rows × cols` matrix.
    #[must_use]
    pub const fn grid_dims(&self, rows: u32, cols: u32) -> (u32, u32) {
        let gx = cols.div_ceil(self.tile_dim);
        let gy = rows.div_ceil(self.tile_dim);
        (gx, gy)
    }

    /// Shared memory size in bytes (f32 elements) including a +1
    /// padding column to avoid bank conflicts.
    #[must_use]
    pub const fn shared_mem_bytes(&self) -> u32 {
        // tile[tile_dim][tile_dim + 1] of f32
        self.tile_dim * (self.tile_dim + 1) * 4
    }
}

impl Default for TileConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}
