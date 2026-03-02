//! CUDA cooperative groups for BitNet kernel launches.
//!
//! Provides Rust-side representations of CUDA cooperative group types, synchronization
//! primitives, and collective operations. These types model the cooperative groups API
//! for configuring multi-block cooperative launches and intra-group communication.
//!
//! All public items are feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to cooperative group operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoopGroupError {
    /// The requested tile size is not a power of two or exceeds warp size.
    InvalidTileSize(u32),
    /// Thread index is out of range for the group.
    ThreadOutOfRange { index: u32, group_size: u32 },
    /// The device does not support cooperative launches.
    CooperativeLaunchUnsupported,
    /// Grid dimensions exceed device limits.
    GridDimensionExceeded { requested: u64, max: u64 },
    /// A synchronization barrier timed out.
    BarrierTimeout,
    /// Reduction operation received an empty input.
    EmptyReduction,
    /// Shuffle source lane is out of range.
    ShuffleLaneOutOfRange { lane: u32, width: u32 },
}

impl fmt::Display for CoopGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTileSize(sz) => {
                write!(f, "invalid tile size {sz}: must be a power of two in [1, 32]")
            }
            Self::ThreadOutOfRange { index, group_size } => {
                write!(f, "thread index {index} out of range for group size {group_size}")
            }
            Self::CooperativeLaunchUnsupported => {
                write!(f, "device does not support cooperative kernel launches")
            }
            Self::GridDimensionExceeded { requested, max } => {
                write!(f, "grid dimension {requested} exceeds device max {max}")
            }
            Self::BarrierTimeout => write!(f, "barrier synchronization timed out"),
            Self::EmptyReduction => write!(f, "reduction on an empty group"),
            Self::ShuffleLaneOutOfRange { lane, width } => {
                write!(f, "shuffle lane {lane} out of range for width {width}")
            }
        }
    }
}

impl std::error::Error for CoopGroupError {}

/// Convenience result alias.
pub type CoopResult<T> = std::result::Result<T, CoopGroupError>;

// ---------------------------------------------------------------------------
// Reduction operation
// ---------------------------------------------------------------------------

/// Reduction operations supported by group collectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    And,
    Or,
    Xor,
}

impl ReduceOp {
    /// Apply the operation to two `f32` values (bitwise ops use bit patterns).
    #[inline]
    pub fn apply_f32(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Sum => a + b,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::And => f32::from_bits(a.to_bits() & b.to_bits()),
            Self::Or => f32::from_bits(a.to_bits() | b.to_bits()),
            Self::Xor => f32::from_bits(a.to_bits() ^ b.to_bits()),
        }
    }

    /// Apply the operation to two `u32` values.
    #[inline]
    pub fn apply_u32(self, a: u32, b: u32) -> u32 {
        match self {
            Self::Sum => a.wrapping_add(b),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
        }
    }
}

// ---------------------------------------------------------------------------
// Core group types
// ---------------------------------------------------------------------------

/// A thread block cooperative group (corresponds to `thread_block` in CUDA).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadBlock {
    block_dim: (u32, u32, u32),
    block_idx: (u32, u32, u32),
}

impl ThreadBlock {
    /// Create a new thread block descriptor.
    pub fn new(block_dim: (u32, u32, u32), block_idx: (u32, u32, u32)) -> CoopResult<Self> {
        if block_dim.0 == 0 || block_dim.1 == 0 || block_dim.2 == 0 {
            return Err(CoopGroupError::InvalidTileSize(0));
        }
        Ok(Self { block_dim, block_idx })
    }

    /// Total number of threads in this block.
    #[inline]
    pub fn num_threads(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    /// Block dimensions `(x, y, z)`.
    #[inline]
    pub fn dim(&self) -> (u32, u32, u32) {
        self.block_dim
    }

    /// Block index within the grid.
    #[inline]
    pub fn index(&self) -> (u32, u32, u32) {
        self.block_idx
    }

    /// Validate that `thread_rank` falls inside this block.
    pub fn validate_thread(&self, thread_rank: u32) -> CoopResult<()> {
        if thread_rank >= self.num_threads() {
            Err(CoopGroupError::ThreadOutOfRange {
                index: thread_rank,
                group_size: self.num_threads(),
            })
        } else {
            Ok(())
        }
    }

    /// Simulate a `__syncthreads()` barrier (returns the set of participating threads).
    pub fn sync(&self) -> SyncToken {
        SyncToken { participating_threads: self.num_threads() }
    }
}

/// A grid-level cooperative group (corresponds to `grid_group` in CUDA).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridGroup {
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
}

impl GridGroup {
    /// Create a grid group from grid and block dimensions.
    pub fn new(grid_dim: (u32, u32, u32), block_dim: (u32, u32, u32)) -> CoopResult<Self> {
        if grid_dim.0 == 0 || grid_dim.1 == 0 || grid_dim.2 == 0 {
            return Err(CoopGroupError::GridDimensionExceeded { requested: 0, max: u64::MAX });
        }
        if block_dim.0 == 0 || block_dim.1 == 0 || block_dim.2 == 0 {
            return Err(CoopGroupError::InvalidTileSize(0));
        }
        Ok(Self { grid_dim, block_dim })
    }

    /// Total number of threads across all blocks.
    pub fn num_threads(&self) -> u64 {
        let blocks = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let threads_per_block =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        blocks * threads_per_block
    }

    /// Number of blocks in the grid.
    pub fn num_blocks(&self) -> u64 {
        self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64
    }

    /// Grid dimensions.
    #[inline]
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        self.grid_dim
    }

    /// Block dimensions.
    #[inline]
    pub fn block_dim(&self) -> (u32, u32, u32) {
        self.block_dim
    }

    /// Simulate a grid-wide barrier.
    pub fn sync(&self) -> SyncToken {
        SyncToken { participating_threads: self.num_threads() as u32 }
    }
}

/// Allowed tile sizes for tiled partitions (must be power of two, ≤ 32).
const VALID_TILE_SIZES: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// A tiled partition of a warp (corresponds to `thread_block_tile<N>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TiledPartition {
    tile_size: u32,
}

impl TiledPartition {
    /// Create a tiled partition. `tile_size` must be 1, 2, 4, 8, 16, or 32.
    pub fn new(tile_size: u32) -> CoopResult<Self> {
        if !VALID_TILE_SIZES.contains(&tile_size) {
            return Err(CoopGroupError::InvalidTileSize(tile_size));
        }
        Ok(Self { tile_size })
    }

    /// Tile size (number of threads in this partition).
    #[inline]
    pub fn size(&self) -> u32 {
        self.tile_size
    }

    /// Simulate sync within the tile.
    pub fn sync(&self) -> SyncToken {
        SyncToken { participating_threads: self.tile_size }
    }

    /// Simulate `__shfl_sync` within the tile.
    pub fn shfl(&self, values: &[f32], src_lane: u32) -> CoopResult<Vec<f32>> {
        if src_lane >= self.tile_size {
            return Err(CoopGroupError::ShuffleLaneOutOfRange {
                lane: src_lane,
                width: self.tile_size,
            });
        }
        if values.len() != self.tile_size as usize {
            return Err(CoopGroupError::ThreadOutOfRange {
                index: values.len() as u32,
                group_size: self.tile_size,
            });
        }
        let broadcast = values[src_lane as usize];
        Ok(vec![broadcast; self.tile_size as usize])
    }

    /// Simulate `__shfl_xor_sync` within the tile.
    pub fn shfl_xor(&self, values: &[f32], mask: u32) -> CoopResult<Vec<f32>> {
        if values.len() != self.tile_size as usize {
            return Err(CoopGroupError::ThreadOutOfRange {
                index: values.len() as u32,
                group_size: self.tile_size,
            });
        }
        let mut result = vec![0.0f32; self.tile_size as usize];
        for lane in 0..self.tile_size {
            let src = lane ^ mask;
            let src = if src < self.tile_size { src } else { lane };
            result[lane as usize] = values[src as usize];
        }
        Ok(result)
    }

    /// Simulate `__ballot_sync` — returns a bitmask of lanes where `predicate[lane]`
    /// is true.
    pub fn ballot(&self, predicates: &[bool]) -> CoopResult<u32> {
        if predicates.len() != self.tile_size as usize {
            return Err(CoopGroupError::ThreadOutOfRange {
                index: predicates.len() as u32,
                group_size: self.tile_size,
            });
        }
        let mut mask = 0u32;
        for (i, &p) in predicates.iter().enumerate() {
            if p {
                mask |= 1 << i;
            }
        }
        Ok(mask)
    }

    /// Simulate `__any_sync` — returns `true` if any lane's predicate is true.
    pub fn any(&self, predicates: &[bool]) -> CoopResult<bool> {
        self.ballot(predicates).map(|m| m != 0)
    }

    /// Simulate `__all_sync` — returns `true` if all lanes' predicates are true.
    pub fn all(&self, predicates: &[bool]) -> CoopResult<bool> {
        let full_mask = (1u32 << self.tile_size) - 1;
        self.ballot(predicates).map(|m| m == full_mask)
    }

    /// Tree-reduce `values` within the tile using `op`.
    pub fn reduce(&self, values: &[f32], op: ReduceOp) -> CoopResult<f32> {
        if values.is_empty() {
            return Err(CoopGroupError::EmptyReduction);
        }
        if values.len() != self.tile_size as usize {
            return Err(CoopGroupError::ThreadOutOfRange {
                index: values.len() as u32,
                group_size: self.tile_size,
            });
        }
        let result = values.iter().copied().reduce(|a, b| op.apply_f32(a, b)).unwrap();
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Synchronisation token
// ---------------------------------------------------------------------------

/// Token returned by `sync()` methods to prove a barrier was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncToken {
    /// How many threads participated.
    pub participating_threads: u32,
}

// ---------------------------------------------------------------------------
// Cooperative launch configuration
// ---------------------------------------------------------------------------

/// Device properties relevant to cooperative launch decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProperties {
    pub max_threads_per_block: u32,
    pub max_blocks_per_sm: u32,
    pub sm_count: u32,
    pub cooperative_launch: bool,
    pub max_grid_dim: (u32, u32, u32),
    pub warp_size: u32,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_blocks_per_sm: 16,
            sm_count: 1,
            cooperative_launch: true,
            max_grid_dim: (2_147_483_647, 65535, 65535),
            warp_size: 32,
        }
    }
}

/// Configuration for a cooperative kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoopLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub cooperative: bool,
}

impl CoopLaunchConfig {
    /// Validate and build a launch configuration.
    pub fn new(
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        device: &DeviceProperties,
    ) -> CoopResult<Self> {
        if !device.cooperative_launch {
            return Err(CoopGroupError::CooperativeLaunchUnsupported);
        }

        let threads_per_block = block_dim.0 as u64 * block_dim.1 as u64 * block_dim.2 as u64;
        if threads_per_block > device.max_threads_per_block as u64 {
            return Err(CoopGroupError::GridDimensionExceeded {
                requested: threads_per_block,
                max: device.max_threads_per_block as u64,
            });
        }

        if grid_dim.0 > device.max_grid_dim.0
            || grid_dim.1 > device.max_grid_dim.1
            || grid_dim.2 > device.max_grid_dim.2
        {
            let requested = grid_dim.0 as u64 * grid_dim.1 as u64 * grid_dim.2 as u64;
            let max = device.max_grid_dim.0 as u64
                * device.max_grid_dim.1 as u64
                * device.max_grid_dim.2 as u64;
            return Err(CoopGroupError::GridDimensionExceeded { requested, max });
        }

        Ok(Self { grid_dim, block_dim, shared_mem_bytes, cooperative: true })
    }

    /// Total number of threads that will participate.
    pub fn total_threads(&self) -> u64 {
        let blocks = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let tpb = self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        blocks * tpb
    }

    /// Maximum number of cooperative blocks the device can run concurrently.
    pub fn max_active_blocks(device: &DeviceProperties, threads_per_block: u32) -> u32 {
        if threads_per_block == 0 {
            return 0;
        }
        let blocks_per_sm =
            device.max_threads_per_block.min(threads_per_block * device.max_blocks_per_sm)
                / threads_per_block;
        blocks_per_sm * device.sm_count
    }
}

// ---------------------------------------------------------------------------
// Helpers: partition a thread block into tiles
// ---------------------------------------------------------------------------

/// Partition a [`ThreadBlock`] into tiled partitions of the given size.
pub fn partition_into_tiles(
    block: &ThreadBlock,
    tile_size: u32,
) -> CoopResult<Vec<TiledPartition>> {
    let tile = TiledPartition::new(tile_size)?;
    let n = block.num_threads();
    if tile_size > n {
        return Err(CoopGroupError::InvalidTileSize(tile_size));
    }
    let count = n / tile_size;
    Ok(vec![tile; count as usize])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ThreadBlock tests --------------------------------------------------

    #[test]
    fn thread_block_basic() {
        let tb = ThreadBlock::new((256, 1, 1), (0, 0, 0)).unwrap();
        assert_eq!(tb.num_threads(), 256);
        assert_eq!(tb.dim(), (256, 1, 1));
        assert_eq!(tb.index(), (0, 0, 0));
    }

    #[test]
    fn thread_block_3d() {
        let tb = ThreadBlock::new((8, 8, 4), (1, 2, 3)).unwrap();
        assert_eq!(tb.num_threads(), 256);
        assert_eq!(tb.index(), (1, 2, 3));
    }

    #[test]
    fn thread_block_zero_dim_rejected() {
        assert!(ThreadBlock::new((0, 1, 1), (0, 0, 0)).is_err());
        assert!(ThreadBlock::new((1, 0, 1), (0, 0, 0)).is_err());
        assert!(ThreadBlock::new((1, 1, 0), (0, 0, 0)).is_err());
    }

    #[test]
    fn thread_block_validate_thread_ok() {
        let tb = ThreadBlock::new((64, 1, 1), (0, 0, 0)).unwrap();
        assert!(tb.validate_thread(0).is_ok());
        assert!(tb.validate_thread(63).is_ok());
    }

    #[test]
    fn thread_block_validate_thread_out_of_range() {
        let tb = ThreadBlock::new((64, 1, 1), (0, 0, 0)).unwrap();
        assert_eq!(
            tb.validate_thread(64),
            Err(CoopGroupError::ThreadOutOfRange { index: 64, group_size: 64 })
        );
    }

    #[test]
    fn thread_block_sync_token() {
        let tb = ThreadBlock::new((128, 1, 1), (0, 0, 0)).unwrap();
        let tok = tb.sync();
        assert_eq!(tok.participating_threads, 128);
    }

    // -- GridGroup tests ----------------------------------------------------

    #[test]
    fn grid_group_basic() {
        let gg = GridGroup::new((4, 1, 1), (256, 1, 1)).unwrap();
        assert_eq!(gg.num_threads(), 1024);
        assert_eq!(gg.num_blocks(), 4);
    }

    #[test]
    fn grid_group_3d() {
        let gg = GridGroup::new((2, 3, 4), (8, 8, 4)).unwrap();
        assert_eq!(gg.num_blocks(), 24);
        assert_eq!(gg.num_threads(), 24 * 256);
    }

    #[test]
    fn grid_group_zero_grid_rejected() {
        assert!(GridGroup::new((0, 1, 1), (256, 1, 1)).is_err());
    }

    #[test]
    fn grid_group_zero_block_rejected() {
        assert!(GridGroup::new((1, 1, 1), (0, 1, 1)).is_err());
    }

    #[test]
    fn grid_group_accessors() {
        let gg = GridGroup::new((2, 2, 2), (32, 1, 1)).unwrap();
        assert_eq!(gg.grid_dim(), (2, 2, 2));
        assert_eq!(gg.block_dim(), (32, 1, 1));
    }

    #[test]
    fn grid_group_sync_token() {
        let gg = GridGroup::new((2, 1, 1), (32, 1, 1)).unwrap();
        let tok = gg.sync();
        assert_eq!(tok.participating_threads, 64);
    }

    // -- TiledPartition tests -----------------------------------------------

    #[test]
    fn tiled_partition_valid_sizes() {
        for &sz in &VALID_TILE_SIZES {
            let tp = TiledPartition::new(sz).unwrap();
            assert_eq!(tp.size(), sz);
        }
    }

    #[test]
    fn tiled_partition_invalid_size_rejected() {
        assert!(TiledPartition::new(0).is_err());
        assert!(TiledPartition::new(3).is_err());
        assert!(TiledPartition::new(5).is_err());
        assert!(TiledPartition::new(64).is_err());
    }

    #[test]
    fn tiled_partition_sync() {
        let tp = TiledPartition::new(16).unwrap();
        assert_eq!(tp.sync().participating_threads, 16);
    }

    #[test]
    fn tiled_partition_shfl_broadcast() {
        let tp = TiledPartition::new(4).unwrap();
        let vals = vec![10.0, 20.0, 30.0, 40.0];
        let out = tp.shfl(&vals, 2).unwrap();
        assert_eq!(out, vec![30.0; 4]);
    }

    #[test]
    fn tiled_partition_shfl_lane_out_of_range() {
        let tp = TiledPartition::new(4).unwrap();
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        assert!(matches!(
            tp.shfl(&vals, 4),
            Err(CoopGroupError::ShuffleLaneOutOfRange { lane: 4, width: 4 })
        ));
    }

    #[test]
    fn tiled_partition_shfl_xor_swap_pairs() {
        let tp = TiledPartition::new(4).unwrap();
        let vals = vec![0.0, 1.0, 2.0, 3.0];
        let out = tp.shfl_xor(&vals, 1).unwrap();
        assert_eq!(out, vec![1.0, 0.0, 3.0, 2.0]);
    }

    #[test]
    fn tiled_partition_shfl_xor_identity() {
        let tp = TiledPartition::new(4).unwrap();
        let vals = vec![10.0, 20.0, 30.0, 40.0];
        let out = tp.shfl_xor(&vals, 0).unwrap();
        assert_eq!(out, vals);
    }

    #[test]
    fn tiled_partition_ballot_all_true() {
        let tp = TiledPartition::new(4).unwrap();
        let mask = tp.ballot(&[true, true, true, true]).unwrap();
        assert_eq!(mask, 0b1111);
    }

    #[test]
    fn tiled_partition_ballot_none_true() {
        let tp = TiledPartition::new(4).unwrap();
        let mask = tp.ballot(&[false, false, false, false]).unwrap();
        assert_eq!(mask, 0);
    }

    #[test]
    fn tiled_partition_ballot_some() {
        let tp = TiledPartition::new(4).unwrap();
        let mask = tp.ballot(&[true, false, true, false]).unwrap();
        assert_eq!(mask, 0b0101);
    }

    #[test]
    fn tiled_partition_any_true() {
        let tp = TiledPartition::new(4).unwrap();
        assert!(tp.any(&[false, false, true, false]).unwrap());
    }

    #[test]
    fn tiled_partition_any_false() {
        let tp = TiledPartition::new(4).unwrap();
        assert!(!tp.any(&[false, false, false, false]).unwrap());
    }

    #[test]
    fn tiled_partition_all_true() {
        let tp = TiledPartition::new(4).unwrap();
        assert!(tp.all(&[true, true, true, true]).unwrap());
    }

    #[test]
    fn tiled_partition_all_false_with_one_false() {
        let tp = TiledPartition::new(4).unwrap();
        assert!(!tp.all(&[true, true, false, true]).unwrap());
    }

    // -- Reduce tests -------------------------------------------------------

    #[test]
    fn reduce_sum() {
        let tp = TiledPartition::new(4).unwrap();
        let sum = tp.reduce(&[1.0, 2.0, 3.0, 4.0], ReduceOp::Sum).unwrap();
        assert!((sum - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reduce_min() {
        let tp = TiledPartition::new(4).unwrap();
        let min = tp.reduce(&[3.0, 1.0, 4.0, 2.0], ReduceOp::Min).unwrap();
        assert!((min - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reduce_max() {
        let tp = TiledPartition::new(4).unwrap();
        let max = tp.reduce(&[3.0, 1.0, 4.0, 2.0], ReduceOp::Max).unwrap();
        assert!((max - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reduce_empty_error() {
        let tp = TiledPartition::new(1).unwrap();
        assert!(matches!(tp.reduce(&[], ReduceOp::Sum), Err(CoopGroupError::EmptyReduction)));
    }

    // -- ReduceOp unit tests ------------------------------------------------

    #[test]
    fn reduce_op_u32_sum() {
        assert_eq!(ReduceOp::Sum.apply_u32(3, 7), 10);
    }

    #[test]
    fn reduce_op_u32_bitwise() {
        assert_eq!(ReduceOp::And.apply_u32(0b1100, 0b1010), 0b1000);
        assert_eq!(ReduceOp::Or.apply_u32(0b1100, 0b1010), 0b1110);
        assert_eq!(ReduceOp::Xor.apply_u32(0b1100, 0b1010), 0b0110);
    }

    // -- CoopLaunchConfig tests ---------------------------------------------

    #[test]
    fn coop_launch_config_valid() {
        let dev = DeviceProperties::default();
        let cfg = CoopLaunchConfig::new((4, 1, 1), (256, 1, 1), 0, &dev).unwrap();
        assert_eq!(cfg.total_threads(), 1024);
        assert!(cfg.cooperative);
    }

    #[test]
    fn coop_launch_config_too_many_threads_per_block() {
        let dev = DeviceProperties { max_threads_per_block: 512, ..Default::default() };
        let res = CoopLaunchConfig::new((1, 1, 1), (1024, 1, 1), 0, &dev);
        assert!(matches!(res, Err(CoopGroupError::GridDimensionExceeded { .. })));
    }

    #[test]
    fn coop_launch_config_unsupported_device() {
        let dev = DeviceProperties { cooperative_launch: false, ..Default::default() };
        let res = CoopLaunchConfig::new((1, 1, 1), (256, 1, 1), 0, &dev);
        assert!(matches!(res, Err(CoopGroupError::CooperativeLaunchUnsupported)));
    }

    #[test]
    fn coop_launch_max_active_blocks() {
        let dev = DeviceProperties {
            max_threads_per_block: 1024,
            max_blocks_per_sm: 16,
            sm_count: 80,
            ..Default::default()
        };
        let max = CoopLaunchConfig::max_active_blocks(&dev, 256);
        assert!(max > 0);
    }

    #[test]
    fn coop_launch_max_active_blocks_zero_threads() {
        let dev = DeviceProperties::default();
        assert_eq!(CoopLaunchConfig::max_active_blocks(&dev, 0), 0);
    }

    // -- Partition helper tests ---------------------------------------------

    #[test]
    fn partition_into_tiles_basic() {
        let tb = ThreadBlock::new((128, 1, 1), (0, 0, 0)).unwrap();
        let tiles = partition_into_tiles(&tb, 32).unwrap();
        assert_eq!(tiles.len(), 4);
        assert!(tiles.iter().all(|t| t.size() == 32));
    }

    #[test]
    fn partition_into_tiles_tile_too_large() {
        let tb = ThreadBlock::new((16, 1, 1), (0, 0, 0)).unwrap();
        assert!(partition_into_tiles(&tb, 32).is_err());
    }

    #[test]
    fn partition_into_tiles_invalid_size() {
        let tb = ThreadBlock::new((128, 1, 1), (0, 0, 0)).unwrap();
        assert!(partition_into_tiles(&tb, 7).is_err());
    }

    // -- Error display tests ------------------------------------------------

    #[test]
    fn error_display_messages() {
        let e = CoopGroupError::InvalidTileSize(5);
        assert!(e.to_string().contains("5"));
        let e = CoopGroupError::BarrierTimeout;
        assert!(e.to_string().contains("timed out"));
        let e = CoopGroupError::EmptyReduction;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CoopGroupError>();
    }

    // -- DeviceProperties default -------------------------------------------

    #[test]
    fn device_properties_default_sane() {
        let dp = DeviceProperties::default();
        assert_eq!(dp.warp_size, 32);
        assert!(dp.cooperative_launch);
        assert!(dp.max_threads_per_block >= 256);
    }

    // -- proptest -----------------------------------------------------------

    mod proptests {
        use super::super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn tile_size_power_of_two_accepted(exp in 0u32..6) {
                let size = 1u32 << exp;
                if size <= 32 {
                    prop_assert!(TiledPartition::new(size).is_ok());
                }
            }

            #[test]
            fn non_power_of_two_tile_rejected(size in 3u32..1000) {
                if !size.is_power_of_two() || size > 32 {
                    prop_assert!(TiledPartition::new(size).is_err());
                }
            }

            #[test]
            fn ballot_popcount_matches(bits in 0u32..16) {
                let tp = TiledPartition::new(4).unwrap();
                let preds: Vec<bool> = (0..4).map(|i| bits & (1 << i) != 0).collect();
                let mask = tp.ballot(&preds).unwrap();
                prop_assert_eq!(mask.count_ones(), preds.iter().filter(|&&p| p).count() as u32);
            }

            #[test]
            fn reduce_sum_matches_iter(a in -100.0f32..100.0, b in -100.0f32..100.0,
                                        c in -100.0f32..100.0, d in -100.0f32..100.0) {
                let tp = TiledPartition::new(4).unwrap();
                let vals = [a, b, c, d];
                let got = tp.reduce(&vals, ReduceOp::Sum).unwrap();
                let expected = a + b + c + d;
                prop_assert!((got - expected).abs() < 1e-4,
                    "got {got}, expected {expected}");
            }

            #[test]
            fn shfl_xor_is_involution(
                a in -100.0f32..100.0, b in -100.0f32..100.0,
                c in -100.0f32..100.0, d in -100.0f32..100.0,
                mask in 0u32..4,
            ) {
                let tp = TiledPartition::new(4).unwrap();
                let vals = vec![a, b, c, d];
                let once = tp.shfl_xor(&vals, mask).unwrap();
                let twice = tp.shfl_xor(&once, mask).unwrap();
                for (orig, round_trip) in vals.iter().zip(twice.iter()) {
                    prop_assert!((orig - round_trip).abs() < f32::EPSILON);
                }
            }

            #[test]
            fn grid_thread_count_consistent(
                gx in 1u32..16, gy in 1u32..8, gz in 1u32..4,
                bx in 1u32..33, by in 1u32..9, bz in 1u32..5,
            ) {
                let gg = GridGroup::new((gx, gy, gz), (bx, by, bz)).unwrap();
                let expected = gx as u64 * gy as u64 * gz as u64
                             * bx as u64 * by as u64 * bz as u64;
                prop_assert_eq!(gg.num_threads(), expected);
            }
        }
    }
}
