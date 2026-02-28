//! GPU workgroup size optimization.
//!
//! Determines optimal local/global work sizes for GPU kernel
//! launches based on device constraints and kernel requirements.

/// Device workgroup constraints queried from the GPU driver.
#[derive(Debug, Clone)]
pub struct DeviceConstraints {
    /// Maximum total work-items in a single work-group.
    pub max_work_group_size: usize,
    /// Per-dimension maximum work-item counts.
    pub max_work_item_sizes: [usize; 3],
    /// Preferred work-group size multiple (warp/wavefront granularity).
    pub preferred_work_group_multiple: usize,
    /// Number of compute units (SMs / EUs / CUs).
    pub compute_units: usize,
    /// Maximum local (shared) memory in bytes.
    pub max_local_memory: usize,
    /// Hardware warp/wavefront width.
    /// 32 for NVIDIA, 64 for AMD, varies for Intel.
    pub warp_size: usize,
}

impl DeviceConstraints {
    /// Defaults for Intel Arc GPUs (Xe-HPG, 16 EUs per slice).
    #[must_use]
    pub const fn intel_arc() -> Self {
        Self {
            max_work_group_size: 1024,
            max_work_item_sizes: [1024, 1024, 64],
            preferred_work_group_multiple: 32,
            compute_units: 512,
            max_local_memory: 65536,
            warp_size: 32,
        }
    }

    /// Defaults for NVIDIA GPUs (SM-based, warp size 32).
    #[must_use]
    pub const fn nvidia() -> Self {
        Self {
            max_work_group_size: 1024,
            max_work_item_sizes: [1024, 1024, 64],
            preferred_work_group_multiple: 32,
            compute_units: 128,
            max_local_memory: 49152,
            warp_size: 32,
        }
    }

    /// Defaults for AMD GPUs (wavefront 64).
    #[must_use]
    pub const fn amd() -> Self {
        Self {
            max_work_group_size: 1024,
            max_work_item_sizes: [1024, 1024, 1024],
            preferred_work_group_multiple: 64,
            compute_units: 120,
            max_local_memory: 65536,
            warp_size: 64,
        }
    }

    /// Generic fallback with conservative limits.
    #[must_use]
    pub const fn generic() -> Self {
        Self {
            max_work_group_size: 256,
            max_work_item_sizes: [256, 256, 64],
            preferred_work_group_multiple: 32,
            compute_units: 16,
            max_local_memory: 32768,
            warp_size: 32,
        }
    }
}

/// A workgroup size recommendation (local + global dimensions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkgroupSize {
    /// Local work-group size per dimension.
    pub local: [usize; 3],
    /// Global work size per dimension (padded to multiples of local).
    pub global: [usize; 3],
}

impl WorkgroupSize {
    /// Total number of work-items across all global dimensions.
    #[must_use]
    pub const fn total_work_items(&self) -> usize {
        self.global[0] * self.global[1] * self.global[2]
    }

    /// Total work-items per work-group (product of local sizes).
    #[must_use]
    pub const fn local_size(&self) -> usize {
        self.local[0] * self.local[1] * self.local[2]
    }
}

/// Integer square root (floor).
const fn isqrt(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = usize::midpoint(x, n / x);
    }
    x
}

/// Optimises workgroup sizes for GPU kernel launches.
///
/// Given [`DeviceConstraints`], selects local and global sizes that:
/// - respect device maximums,
/// - align to the preferred work-group multiple,
/// - pad global sizes so every work-item is covered.
pub struct WorkgroupOptimizer {
    constraints: DeviceConstraints,
}

impl WorkgroupOptimizer {
    /// Create an optimizer from the given device constraints.
    #[must_use]
    pub const fn new(constraints: DeviceConstraints) -> Self {
        Self { constraints }
    }

    /// Optimise for a 1-D kernel (e.g. element-wise ops).
    #[must_use]
    pub const fn optimize_1d(&self, total_items: usize) -> WorkgroupSize {
        let c = &self.constraints;
        // Start from the preferred multiple, grow by multiples while
        // staying within both the total and per-dim limits.
        let mut local = c.preferred_work_group_multiple;
        while local * 2 <= c.max_work_group_size
            && local * 2 <= c.max_work_item_sizes[0]
            && local * 2 <= total_items
        {
            local *= 2;
        }
        let global = Self::round_up(total_items, local);
        WorkgroupSize { local: [local, 1, 1], global: [global, 1, 1] }
    }

    /// Optimise for a 2-D kernel (e.g. matrix multiply).
    #[must_use]
    pub fn optimize_2d(&self, rows: usize, cols: usize) -> WorkgroupSize {
        let c = &self.constraints;
        // Square-ish local block whose product ≤ max_work_group_size.
        let mut lx = c.preferred_work_group_multiple.min(c.max_work_item_sizes[0]);
        let mut ly: usize = 1;

        // Grow ly while total stays within limits.
        while ly * 2 <= c.max_work_item_sizes[1] && lx * (ly * 2) <= c.max_work_group_size {
            ly *= 2;
        }

        // Shrink lx if it exceeds rows, keeping alignment.
        while lx > c.preferred_work_group_multiple && lx > rows {
            lx /= 2;
        }

        let gx = Self::round_up(cols, lx);
        let gy = Self::round_up(rows, ly);
        WorkgroupSize { local: [lx, ly, 1], global: [gx, gy, 1] }
    }

    /// Optimise for a tiled matrix multiply (M × N with fixed tile size).
    #[must_use]
    pub fn optimize_tiled_matmul(&self, m: usize, n: usize, tile_size: usize) -> WorkgroupSize {
        let c = &self.constraints;
        let tile = tile_size.min(c.max_work_item_sizes[0]).min(c.max_work_item_sizes[1]);

        // Ensure local product does not exceed max_work_group_size.
        let (lx, ly) = if tile * tile > c.max_work_group_size {
            // Integer square-root, rounded down to nearest power-of-two.
            let side = isqrt(c.max_work_group_size).next_power_of_two() / 2;
            (side.max(1), side.max(1))
        } else {
            (tile, tile)
        };

        let tiles_n = Self::round_up(n, lx);
        let tiles_m = Self::round_up(m, ly);
        WorkgroupSize { local: [lx, ly, 1], global: [tiles_n, tiles_m, 1] }
    }

    /// Optimise for a reduction kernel (e.g. softmax, layer-norm).
    ///
    /// Each row is reduced independently; the local size in x spans
    /// as much of the row as possible (up to one warp per row).
    #[must_use]
    pub const fn optimize_reduction(&self, rows: usize, cols: usize) -> WorkgroupSize {
        let c = &self.constraints;

        // Local x: cover the reduction dimension, clamped.
        let mut lx = c.preferred_work_group_multiple;
        while lx * 2 <= c.max_work_group_size
            && lx * 2 <= c.max_work_item_sizes[0]
            && lx * 2 <= cols
        {
            lx *= 2;
        }

        let gx = Self::round_up(cols, lx);
        let gy = rows;
        WorkgroupSize { local: [lx, 1, 1], global: [gx, gy, 1] }
    }

    /// Round `value` up to the next multiple of `multiple`.
    const fn round_up(value: usize, multiple: usize) -> usize {
        if multiple == 0 {
            return value;
        }
        let remainder = value % multiple;
        if remainder == 0 { value } else { value + (multiple - remainder) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeviceConstraints defaults ──────────────────────────────

    #[test]
    fn intel_arc_defaults_are_valid() {
        let c = DeviceConstraints::intel_arc();
        assert_eq!(c.max_work_group_size, 1024);
        assert_eq!(c.warp_size, 32);
        assert_eq!(c.compute_units, 512);
        assert_eq!(c.max_local_memory, 65536);
    }

    #[test]
    fn nvidia_defaults_are_valid() {
        let c = DeviceConstraints::nvidia();
        assert_eq!(c.max_work_group_size, 1024);
        assert_eq!(c.warp_size, 32);
        assert_eq!(c.compute_units, 128);
        assert_eq!(c.max_local_memory, 49152);
    }

    #[test]
    fn amd_defaults_are_valid() {
        let c = DeviceConstraints::amd();
        assert_eq!(c.max_work_group_size, 1024);
        assert_eq!(c.warp_size, 64);
        assert_eq!(c.preferred_work_group_multiple, 64);
        assert_eq!(c.max_local_memory, 65536);
    }

    #[test]
    fn generic_defaults_are_conservative() {
        let c = DeviceConstraints::generic();
        assert_eq!(c.max_work_group_size, 256);
        assert_eq!(c.warp_size, 32);
        assert_eq!(c.compute_units, 16);
        assert_eq!(c.max_local_memory, 32768);
    }

    #[test]
    fn intel_arc_max_work_item_sizes() {
        let c = DeviceConstraints::intel_arc();
        assert_eq!(c.max_work_item_sizes, [1024, 1024, 64]);
    }

    #[test]
    fn amd_max_work_item_sizes() {
        let c = DeviceConstraints::amd();
        assert_eq!(c.max_work_item_sizes, [1024, 1024, 1024]);
    }

    // ── WorkgroupSize helpers ───────────────────────────────────

    #[test]
    fn workgroup_size_total_work_items() {
        let ws = WorkgroupSize { local: [32, 1, 1], global: [1024, 1, 1] };
        assert_eq!(ws.total_work_items(), 1024);
    }

    #[test]
    fn workgroup_size_local_size() {
        let ws = WorkgroupSize { local: [16, 16, 1], global: [256, 256, 1] };
        assert_eq!(ws.local_size(), 256);
    }

    #[test]
    fn workgroup_size_3d_total() {
        let ws = WorkgroupSize { local: [8, 4, 2], global: [64, 32, 16] };
        assert_eq!(ws.total_work_items(), 64 * 32 * 16);
        assert_eq!(ws.local_size(), 64);
    }

    // ── round_up ────────────────────────────────────────────────

    #[test]
    fn round_up_exact_multiple() {
        assert_eq!(WorkgroupOptimizer::round_up(256, 32), 256);
    }

    #[test]
    fn round_up_non_multiple() {
        assert_eq!(WorkgroupOptimizer::round_up(100, 32), 128);
    }

    #[test]
    fn round_up_one() {
        assert_eq!(WorkgroupOptimizer::round_up(1, 32), 32);
    }

    #[test]
    fn round_up_zero_multiple() {
        assert_eq!(WorkgroupOptimizer::round_up(42, 0), 42);
    }

    #[test]
    fn round_up_value_zero() {
        assert_eq!(WorkgroupOptimizer::round_up(0, 64), 0);
    }

    // ── 1-D optimization ────────────────────────────────────────

    #[test]
    fn optimize_1d_small() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_1d(64);
        assert_eq!(ws.local[0], 64);
        assert!(ws.global[0] >= 64);
        assert_eq!(ws.global[0] % ws.local[0], 0);
    }

    #[test]
    fn optimize_1d_large() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_1d(100_000);
        assert!(ws.local[0] >= 32);
        assert!(ws.local[0] <= 1024);
        assert!(ws.global[0] >= 100_000);
        assert_eq!(ws.global[0] % ws.local[0], 0);
    }

    #[test]
    fn optimize_1d_non_power_of_two() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_1d(1000);
        assert!(ws.global[0] >= 1000);
        assert_eq!(ws.global[0] % ws.local[0], 0);
    }

    #[test]
    fn optimize_1d_single_item() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_1d(1);
        assert_eq!(ws.local[0], 32); // preferred multiple
        assert_eq!(ws.global[0], 32);
    }

    #[test]
    fn optimize_1d_generic_device() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::generic());
        let ws = opt.optimize_1d(4096);
        assert!(ws.local[0] <= 256);
        assert!(ws.global[0] >= 4096);
    }

    #[test]
    fn optimize_1d_amd() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::amd());
        let ws = opt.optimize_1d(2048);
        // AMD preferred multiple is 64
        assert_eq!(ws.local[0] % 64, 0);
        assert!(ws.global[0] >= 2048);
    }

    #[test]
    fn optimize_1d_never_exceeds_max() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::generic());
        let ws = opt.optimize_1d(1_000_000);
        assert!(ws.local[0] <= 256);
    }

    // ── 2-D optimization ────────────────────────────────────────

    #[test]
    fn optimize_2d_square() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_2d(1024, 1024);
        assert!(ws.local_size() <= 1024);
        assert!(ws.global[0] >= 1024);
        assert!(ws.global[1] >= 1024);
        assert_eq!(ws.global[0] % ws.local[0], 0);
        assert_eq!(ws.global[1] % ws.local[1], 0);
    }

    #[test]
    fn optimize_2d_rectangular() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_2d(128, 4096);
        assert!(ws.local_size() <= 1024);
        assert!(ws.global[0] >= 4096);
        assert!(ws.global[1] >= 128);
    }

    #[test]
    fn optimize_2d_small_matrix() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_2d(4, 4);
        assert!(ws.local_size() <= 1024);
        assert!(ws.global[0] >= 4);
        assert!(ws.global[1] >= 4);
    }

    #[test]
    fn optimize_2d_single_row() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_2d(1, 2048);
        assert!(ws.global[0] >= 2048);
        assert!(ws.global[1] >= 1);
    }

    #[test]
    fn optimize_2d_never_exceeds_max() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::generic());
        let ws = opt.optimize_2d(2048, 2048);
        assert!(ws.local_size() <= 256);
    }

    // ── Tiled matmul ────────────────────────────────────────────

    #[test]
    fn optimize_tiled_matmul_16() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_tiled_matmul(1024, 1024, 16);
        assert_eq!(ws.local, [16, 16, 1]);
        assert_eq!(ws.global[0], 1024);
        assert_eq!(ws.global[1], 1024);
    }

    #[test]
    fn optimize_tiled_matmul_32() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_tiled_matmul(512, 512, 32);
        assert_eq!(ws.local, [32, 32, 1]);
        assert_eq!(ws.global[0], 512);
        assert_eq!(ws.global[1], 512);
    }

    #[test]
    fn optimize_tiled_matmul_non_aligned() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_tiled_matmul(100, 200, 16);
        assert!(ws.global[0] >= 200);
        assert!(ws.global[1] >= 100);
        assert_eq!(ws.global[0] % ws.local[0], 0);
        assert_eq!(ws.global[1] % ws.local[1], 0);
    }

    #[test]
    fn optimize_tiled_matmul_large_tile_clamped() {
        // Tile 64×64 = 4096 > max 1024 ⇒ should clamp.
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_tiled_matmul(256, 256, 64);
        assert!(ws.local_size() <= 1024);
    }

    #[test]
    fn optimize_tiled_matmul_generic_small_tile() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::generic());
        let ws = opt.optimize_tiled_matmul(512, 512, 8);
        assert_eq!(ws.local, [8, 8, 1]);
        assert!(ws.global[0] >= 512);
    }

    #[test]
    fn optimize_tiled_matmul_single_tile() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_tiled_matmul(8, 8, 16);
        assert_eq!(ws.local, [16, 16, 1]);
        assert_eq!(ws.global[0], 16);
        assert_eq!(ws.global[1], 16);
    }

    // ── Reduction ───────────────────────────────────────────────

    #[test]
    fn optimize_reduction_typical() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_reduction(32, 4096);
        assert!(ws.local[0] >= 32);
        assert_eq!(ws.global[0] % ws.local[0], 0);
        assert_eq!(ws.global[1], 32);
    }

    #[test]
    fn optimize_reduction_small_cols() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_reduction(16, 64);
        assert!(ws.local[0] <= 64);
        assert!(ws.global[0] >= 64);
        assert_eq!(ws.global[1], 16);
    }

    #[test]
    fn optimize_reduction_single_row() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_reduction(1, 1024);
        assert_eq!(ws.global[1], 1);
        assert!(ws.global[0] >= 1024);
    }

    #[test]
    fn optimize_reduction_amd_wavefront() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::amd());
        let ws = opt.optimize_reduction(64, 2048);
        assert_eq!(ws.local[0] % 64, 0);
        assert!(ws.global[0] >= 2048);
    }

    #[test]
    fn optimize_reduction_never_exceeds_max() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::generic());
        let ws = opt.optimize_reduction(128, 100_000);
        assert!(ws.local[0] <= 256);
    }

    // ── Cross-cutting / edge-case tests ─────────────────────────

    #[test]
    fn global_always_gte_problem_size_1d() {
        for &n in &[1, 7, 33, 127, 1000, 65537] {
            let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
            let ws = opt.optimize_1d(n);
            assert!(ws.global[0] >= n, "global[0]={} < n={n}", ws.global[0]);
        }
    }

    #[test]
    fn global_always_gte_problem_size_2d() {
        for &(r, c) in &[(1, 1), (3, 7), (100, 200), (4096, 4096)] {
            let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
            let ws = opt.optimize_2d(r, c);
            assert!(ws.global[0] >= c);
            assert!(ws.global[1] >= r);
        }
    }

    #[test]
    fn local_product_within_max_for_all_vendors() {
        let vendors = [
            DeviceConstraints::intel_arc(),
            DeviceConstraints::nvidia(),
            DeviceConstraints::amd(),
            DeviceConstraints::generic(),
        ];
        for c in &vendors {
            let opt = WorkgroupOptimizer::new(c.clone());
            let ws = opt.optimize_2d(4096, 4096);
            assert!(
                ws.local_size() <= c.max_work_group_size,
                "vendor {:?}: local_size {} > max {}",
                c.warp_size,
                ws.local_size(),
                c.max_work_group_size,
            );
        }
    }

    #[test]
    fn local_dims_within_max_work_item_sizes() {
        let c = DeviceConstraints::nvidia();
        let opt = WorkgroupOptimizer::new(c.clone());
        let ws = opt.optimize_2d(4096, 4096);
        assert!(ws.local[0] <= c.max_work_item_sizes[0]);
        assert!(ws.local[1] <= c.max_work_item_sizes[1]);
        assert!(ws.local[2] <= c.max_work_item_sizes[2]);
    }

    #[test]
    fn reduction_y_equals_rows() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_reduction(42, 1024);
        assert_eq!(ws.global[1], 42);
        assert_eq!(ws.local[1], 1);
    }

    #[test]
    fn optimize_1d_intel_arc() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::intel_arc());
        let ws = opt.optimize_1d(8192);
        assert!(ws.local[0] >= 32);
        assert!(ws.local[0] <= 1024);
        assert!(ws.global[0] >= 8192);
    }

    #[test]
    fn tiled_matmul_global_divisible_by_local() {
        let opt = WorkgroupOptimizer::new(DeviceConstraints::nvidia());
        let ws = opt.optimize_tiled_matmul(300, 500, 16);
        assert_eq!(ws.global[0] % ws.local[0], 0);
        assert_eq!(ws.global[1] % ws.local[1], 0);
    }
}
