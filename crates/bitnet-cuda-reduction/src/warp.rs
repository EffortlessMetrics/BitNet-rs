//! Warp-level shuffle primitives for efficient parallel reductions.
//!
//! On real CUDA hardware these map to `__shfl_down_sync` / `__shfl_xor_sync`.
//! The CPU-side emulations here mirror the semantics so that kernel logic can
//! be tested and validated without a GPU.

/// Warp size used by CUDA shuffle primitives.
pub const WARP_SIZE: u32 = 32;

/// Warp-level shuffle primitives that emulate CUDA `__shfl_*_sync` intrinsics.
///
/// On a real GPU these would compile to PTX warp-shuffle instructions.  The
/// CPU implementation operates on a lane array so that the same algorithmic
/// patterns can be unit-tested on the host.
pub struct WarpPrimitives;

impl WarpPrimitives {
    // -- construction helpers ------------------------------------------------

    /// Create a lane array pre-filled with `value` in every lane.
    ///
    /// ```
    /// use bitnet_cuda_reduction::WarpPrimitives;
    ///
    /// let lanes = WarpPrimitives::broadcast(1.0_f32);
    /// assert!(lanes.iter().all(|&v| v == 1.0));
    /// ```
    #[must_use]
    pub const fn broadcast(value: f32) -> [f32; WARP_SIZE as usize] {
        [value; WARP_SIZE as usize]
    }

    // -- shuffle intrinsics --------------------------------------------------

    /// Emulate `__shfl_down_sync`: each lane reads the value `delta` lanes
    /// ahead.  Lanes whose source index exceeds [`WARP_SIZE`] keep their own
    /// value.
    ///
    /// ```
    /// use bitnet_cuda_reduction::WarpPrimitives;
    ///
    /// let mut lanes = [0.0_f32; 32];
    /// lanes[0] = 10.0;
    /// lanes[1] = 20.0;
    /// let result = WarpPrimitives::shfl_down_sync(&lanes, 1);
    /// assert_eq!(result[0], 20.0); // lane 0 reads from lane 1
    /// ```
    #[must_use]
    pub fn shfl_down_sync(
        lanes: &[f32; WARP_SIZE as usize],
        delta: u32,
    ) -> [f32; WARP_SIZE as usize] {
        let mut out = *lanes;
        for i in 0..WARP_SIZE {
            let src = i + delta;
            if src < WARP_SIZE {
                out[i as usize] = lanes[src as usize];
            }
        }
        out
    }

    /// Emulate `__shfl_xor_sync`: each lane reads from the lane whose index
    /// differs by `mask` (bitwise XOR).
    ///
    /// ```
    /// use bitnet_cuda_reduction::WarpPrimitives;
    ///
    /// let mut lanes = [0.0_f32; 32];
    /// lanes[0] = 1.0;
    /// lanes[1] = 2.0;
    /// let result = WarpPrimitives::shfl_xor_sync(&lanes, 1);
    /// assert_eq!(result[0], 2.0); // lane 0 XOR 1 = lane 1
    /// assert_eq!(result[1], 1.0); // lane 1 XOR 1 = lane 0
    /// ```
    #[must_use]
    pub fn shfl_xor_sync(
        lanes: &[f32; WARP_SIZE as usize],
        mask: u32,
    ) -> [f32; WARP_SIZE as usize] {
        let mut out = *lanes;
        for i in 0..WARP_SIZE {
            let src = i ^ mask;
            if src < WARP_SIZE {
                out[i as usize] = lanes[src as usize];
            }
        }
        out
    }

    /// Perform a full warp-level sum reduction using `shfl_down_sync`.
    ///
    /// After the call, `lanes[0]` contains the sum of all 32 lanes.
    ///
    /// ```
    /// use bitnet_cuda_reduction::WarpPrimitives;
    ///
    /// let lanes = [1.0_f32; 32];
    /// let reduced = WarpPrimitives::warp_reduce_sum(&lanes);
    /// assert!((reduced[0] - 32.0).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn warp_reduce_sum(lanes: &[f32; WARP_SIZE as usize]) -> [f32; WARP_SIZE as usize] {
        let mut current = *lanes;
        let mut offset = WARP_SIZE / 2;
        while offset > 0 {
            let shuffled = Self::shfl_down_sync(&current, offset);
            for i in 0..WARP_SIZE as usize {
                current[i] += shuffled[i];
            }
            offset /= 2;
        }
        current
    }

    /// Perform a full warp-level max reduction using `shfl_down_sync`.
    ///
    /// After the call, `lanes[0]` contains the maximum across all 32 lanes.
    ///
    /// ```
    /// use bitnet_cuda_reduction::WarpPrimitives;
    ///
    /// let mut lanes = [0.0_f32; 32];
    /// lanes[7] = 42.0;
    /// let reduced = WarpPrimitives::warp_reduce_max(&lanes);
    /// assert!((reduced[0] - 42.0).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn warp_reduce_max(lanes: &[f32; WARP_SIZE as usize]) -> [f32; WARP_SIZE as usize] {
        let mut current = *lanes;
        let mut offset = WARP_SIZE / 2;
        while offset > 0 {
            let shuffled = Self::shfl_down_sync(&current, offset);
            for i in 0..WARP_SIZE as usize {
                current[i] = f32::max(current[i], shuffled[i]);
            }
            offset /= 2;
        }
        current
    }

    /// Perform a full warp-level min reduction using `shfl_down_sync`.
    #[must_use]
    pub fn warp_reduce_min(lanes: &[f32; WARP_SIZE as usize]) -> [f32; WARP_SIZE as usize] {
        let mut current = *lanes;
        let mut offset = WARP_SIZE / 2;
        while offset > 0 {
            let shuffled = Self::shfl_down_sync(&current, offset);
            for i in 0..WARP_SIZE as usize {
                current[i] = f32::min(current[i], shuffled[i]);
            }
            offset /= 2;
        }
        current
    }
}

// ---------------------------------------------------------------------------
// GPU stubs — compiled only when `gpu` or `cuda` feature is active
// ---------------------------------------------------------------------------

/// GPU-resident warp primitives (requires `gpu` or `cuda` feature).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu {
    use super::WARP_SIZE;

    /// Parameters for launching a warp-level reduction kernel on a GPU.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct WarpLaunchParams {
        /// Number of threads per block (must be a multiple of [`WARP_SIZE`]).
        pub threads_per_block: u32,
        /// Number of blocks in the grid.
        pub grid_size: u32,
        /// Shared memory in bytes required per block.
        pub shared_mem_bytes: u32,
    }

    impl WarpLaunchParams {
        /// Compute launch parameters for reducing `n` elements.
        ///
        /// `threads_per_block` is clamped to the range
        /// `[WARP_SIZE, max_threads]` and rounded up to a warp multiple.
        #[must_use]
        pub fn for_reduction(n: u32, max_threads: u32) -> Self {
            let threads = n.min(max_threads).max(WARP_SIZE);
            let threads = (threads + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
            let grid = (n + threads - 1) / threads;
            let shared = threads * 4; // 4 bytes per f32
            Self { threads_per_block: threads, grid_size: grid.max(1), shared_mem_bytes: shared }
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_fills_all_lanes() {
        let lanes = WarpPrimitives::broadcast(2.71);
        assert!(lanes.iter().all(|&v| (v - 2.71).abs() < f32::EPSILON));
    }

    #[test]
    fn shfl_down_basic() {
        let mut lanes = [0.0_f32; WARP_SIZE as usize];
        for (i, lane) in lanes.iter_mut().enumerate() {
            *lane = i as f32;
        }
        let result = WarpPrimitives::shfl_down_sync(&lanes, 1);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        // last lane keeps its own value (no source beyond warp)
        assert!((result[31] - 31.0).abs() < f32::EPSILON);
    }

    #[test]
    fn shfl_down_zero_delta_is_identity() {
        let lanes = WarpPrimitives::broadcast(7.0);
        let result = WarpPrimitives::shfl_down_sync(&lanes, 0);
        assert_eq!(lanes, result);
    }

    #[test]
    fn shfl_xor_swaps_adjacent_pairs() {
        let mut lanes = [0.0_f32; WARP_SIZE as usize];
        lanes[0] = 10.0;
        lanes[1] = 20.0;
        let result = WarpPrimitives::shfl_xor_sync(&lanes, 1);
        assert!((result[0] - 20.0).abs() < f32::EPSILON);
        assert!((result[1] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn shfl_xor_zero_mask_is_identity() {
        let lanes = WarpPrimitives::broadcast(5.0);
        let result = WarpPrimitives::shfl_xor_sync(&lanes, 0);
        assert_eq!(lanes, result);
    }

    #[test]
    fn warp_reduce_sum_uniform() {
        let lanes = WarpPrimitives::broadcast(1.0);
        let r = WarpPrimitives::warp_reduce_sum(&lanes);
        assert!((r[0] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn warp_reduce_sum_sequential() {
        let mut lanes = [0.0_f32; WARP_SIZE as usize];
        for (i, lane) in lanes.iter_mut().enumerate() {
            *lane = (i + 1) as f32;
        }
        let r = WarpPrimitives::warp_reduce_sum(&lanes);
        let expected: f32 = (1..=32).map(|x| x as f32).sum();
        assert!((r[0] - expected).abs() < 1e-3);
    }

    #[test]
    fn warp_reduce_max_finds_maximum() {
        let mut lanes = WarpPrimitives::broadcast(-1.0);
        lanes[17] = 99.0;
        let r = WarpPrimitives::warp_reduce_max(&lanes);
        assert!((r[0] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn warp_reduce_min_finds_minimum() {
        let mut lanes = WarpPrimitives::broadcast(100.0);
        lanes[5] = -42.0;
        let r = WarpPrimitives::warp_reduce_min(&lanes);
        assert!((r[0] - (-42.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn warp_reduce_sum_all_zeros() {
        let lanes = WarpPrimitives::broadcast(0.0);
        let r = WarpPrimitives::warp_reduce_sum(&lanes);
        assert!((r[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn warp_reduce_max_all_negative() {
        let lanes = WarpPrimitives::broadcast(-5.0);
        let r = WarpPrimitives::warp_reduce_max(&lanes);
        assert!((r[0] - (-5.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn warp_reduce_min_all_same() {
        let lanes = WarpPrimitives::broadcast(7.7);
        let r = WarpPrimitives::warp_reduce_min(&lanes);
        assert!((r[0] - 7.7).abs() < f32::EPSILON);
    }

    #[test]
    fn shfl_down_large_delta_preserves_values() {
        let lanes = WarpPrimitives::broadcast(3.0);
        let result = WarpPrimitives::shfl_down_sync(&lanes, WARP_SIZE);
        // All lanes keep their value since no source exists beyond warp
        assert_eq!(result, lanes);
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    mod gpu_tests {
        use super::super::WARP_SIZE;
        use super::super::gpu::WarpLaunchParams;

        #[test]
        fn launch_params_minimum_warp() {
            let p = WarpLaunchParams::for_reduction(1, 256);
            assert_eq!(p.threads_per_block, WARP_SIZE);
            assert_eq!(p.grid_size, 1);
        }

        #[test]
        fn launch_params_large_n() {
            let p = WarpLaunchParams::for_reduction(10_000, 256);
            assert_eq!(p.threads_per_block, 256);
            assert!(p.grid_size > 1);
        }

        #[test]
        fn launch_params_shared_mem() {
            let p = WarpLaunchParams::for_reduction(128, 128);
            assert_eq!(p.shared_mem_bytes, 128 * 4);
        }
    }
}
