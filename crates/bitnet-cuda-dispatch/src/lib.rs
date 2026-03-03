//! CUDA kernel dispatch planning and launch configuration utilities.
//!
//! This crate provides pure-logic dispatch planning for CUDA kernel launches
//! without depending on any GPU runtime. It computes grid/block dimensions,
//! occupancy-based launch configurations, batch dispatch plans, and work
//! partitioning strategies.
//!
//! All GPU-specific code is gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

mod batch;
mod config;
mod partition;
mod stream;

pub use batch::{BatchEntry, BatchPlan, BatchValidationError};
pub use config::{LaunchConfig, OccupancyHint};
pub use partition::{PartitionStrategy, WorkPartition, partition};
pub use stream::{StreamPlan, StreamSlot, StreamValidationError, TransferDirection};

// ── CUDA hardware constants ────────────────────────────────────────────────

/// Maximum threads per block on all modern CUDA GPUs (compute ≥ 2.0).
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

/// Maximum blocks in the X grid dimension.
pub const MAX_GRID_DIM_X: u32 = 2_147_483_647; // 2^31 - 1

/// Maximum blocks in the Y/Z grid dimensions.
pub const MAX_GRID_DIM_YZ: u32 = 65_535;

/// Warp size on all NVIDIA GPUs.
pub const WARP_SIZE: u32 = 32;

/// Default threads-per-block for element-wise kernels.
pub const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Maximum shared memory per block (48 KiB baseline).
pub const DEFAULT_SHARED_MEM_BYTES: u32 = 49_152;

// ── Dimension helpers ──────────────────────────────────────────────────────

/// Compute 1-D grid blocks needed to cover `n` elements at `block_size`
/// threads per block, clamped to [`MAX_GRID_DIM_X`].
#[must_use]
pub fn grid_blocks_1d(n: u64, block_size: u32) -> u32 {
    assert!(block_size > 0, "block_size must be > 0");
    let blocks = n.div_ceil(u64::from(block_size));
    let clamped = blocks.min(u64::from(MAX_GRID_DIM_X));
    #[allow(clippy::cast_possible_truncation)] // clamped ≤ u32::MAX
    {
        clamped as u32
    }
}

/// Round `n` up to the next multiple of [`WARP_SIZE`].
#[must_use]
pub const fn round_to_warp(n: u32) -> u32 {
    n.div_ceil(WARP_SIZE) * WARP_SIZE
}

/// Clamp a requested block size to valid CUDA bounds (`1..=1024`),
/// rounded up to a warp multiple.
#[must_use]
pub fn clamp_block_size(requested: u32) -> u32 {
    let clamped = requested.clamp(1, MAX_THREADS_PER_BLOCK);
    round_to_warp(clamped).min(MAX_THREADS_PER_BLOCK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_blocks_basic() {
        assert_eq!(grid_blocks_1d(256, 256), 1);
        assert_eq!(grid_blocks_1d(257, 256), 2);
        assert_eq!(grid_blocks_1d(512, 256), 2);
        assert_eq!(grid_blocks_1d(1, 256), 1);
    }

    #[test]
    fn grid_blocks_large() {
        let huge = u64::MAX;
        assert_eq!(grid_blocks_1d(huge, 256), MAX_GRID_DIM_X);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn grid_blocks_zero_block() {
        let _ = grid_blocks_1d(100, 0);
    }

    #[test]
    fn round_to_warp_exact() {
        assert_eq!(round_to_warp(32), 32);
        assert_eq!(round_to_warp(64), 64);
    }

    #[test]
    fn round_to_warp_up() {
        assert_eq!(round_to_warp(1), 32);
        assert_eq!(round_to_warp(33), 64);
        assert_eq!(round_to_warp(63), 64);
    }

    #[test]
    fn clamp_block_size_within_range() {
        assert_eq!(clamp_block_size(256), 256);
        assert_eq!(clamp_block_size(128), 128);
    }

    #[test]
    fn clamp_block_size_too_large() {
        assert_eq!(clamp_block_size(2048), 1024);
    }

    #[test]
    fn clamp_block_size_rounds_up() {
        assert_eq!(clamp_block_size(100), 128);
        assert_eq!(clamp_block_size(1), 32);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn grid_blocks_covers_all_elements(n in 1u64..10_000_000, bs in 1u32..=1024) {
            let blocks = grid_blocks_1d(n, bs);
            let coverage = u64::from(blocks) * u64::from(bs);
            prop_assert!(coverage >= n || blocks == MAX_GRID_DIM_X);
        }

        #[test]
        fn clamp_block_size_always_valid(req in 0u32..=4096) {
            let bs = clamp_block_size(req.max(1));
            prop_assert!(bs >= WARP_SIZE);
            prop_assert!(bs <= MAX_THREADS_PER_BLOCK);
            prop_assert!(bs.is_multiple_of(WARP_SIZE));
        }

        #[test]
        fn round_to_warp_is_multiple(n in 1u32..100_000) {
            let r = round_to_warp(n);
            prop_assert!(r >= n);
            prop_assert!(r.is_multiple_of(WARP_SIZE));
            prop_assert!(r - n < WARP_SIZE);
        }

        #[test]
        fn launch_config_1d_always_valid(n in 1u64..10_000_000) {
            let c = LaunchConfig::for_elements(n);
            prop_assert!(c.is_valid());
            prop_assert!(c.total_threads() >= n || c.grid.0 == MAX_GRID_DIM_X);
        }

        #[test]
        fn launch_config_2d_always_valid(rows in 1u32..10_000, cols in 1u32..10_000) {
            let c = LaunchConfig::for_matrix(rows, cols);
            prop_assert!(c.is_valid());
        }

        #[test]
        fn partition_rr_covers_all(n in 1u64..100_000, bs in 32u32..=1024) {
            let parts = partition(
                n,
                clamp_block_size(bs),
                PartitionStrategy::RoundRobin,
                &[],
            );
            let total: u64 = parts.iter().map(WorkPartition::len).sum();
            prop_assert_eq!(total, n);
        }

        #[test]
        fn partition_rr_contiguous(n in 2u64..50_000, bs in 32u32..=512) {
            let parts = partition(
                n,
                clamp_block_size(bs),
                PartitionStrategy::RoundRobin,
                &[],
            );
            for w in parts.windows(2) {
                prop_assert_eq!(w[0].end, w[1].start);
            }
        }

        #[test]
        fn partition_dynamic_never_exceeds_original(
            n in 1u64..50_000,
            steal in 0.0f64..=1.0,
        ) {
            let parts = partition(
                n,
                DEFAULT_BLOCK_SIZE,
                PartitionStrategy::Dynamic { steal_ratio: steal },
                &[],
            );
            let total: u64 = parts.iter().map(WorkPartition::len).sum();
            prop_assert!(total <= n);
            prop_assert!(parts.iter().all(|p| !p.is_empty()));
        }

        #[test]
        fn batch_plan_total_threads_is_sum(count in 1usize..20) {
            let mut plan = BatchPlan::new();
            let mut expected = 0u64;
            for i in 0..count {
                let n = ((i + 1) * 256) as u64;
                let cfg = LaunchConfig::for_elements(n);
                expected += cfg.total_threads();
                plan.push_simple(format!("k{i}"), cfg);
            }
            prop_assert_eq!(plan.total_threads(), expected);
        }

        #[test]
        fn stream_plan_kernel_count(k in 1usize..30) {
            let mut plan = StreamPlan::new(2);
            for i in 0..k {
                let sid = i % 2;
                plan.push_kernel(sid, format!("k{i}"), LaunchConfig::for_elements(256));
            }
            prop_assert_eq!(plan.total_kernels(), k);
        }

        #[test]
        fn warps_per_block_consistent(n in 1u64..1_000_000) {
            let c = LaunchConfig::for_elements(n);
            let threads_per_block = c.block.0 * c.block.1 * c.block.2;
            let expected_warps = threads_per_block.div_ceil(WARP_SIZE);
            prop_assert_eq!(c.warps_per_block(), expected_warps);
        }
    }
}
