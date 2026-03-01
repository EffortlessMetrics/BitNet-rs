//! OpenCL work size optimization for Intel Arc GPU compute dispatching.
//!
//! Calculates optimal global and local work sizes based on Intel Arc hardware
//! constraints: 16-wide SIMD subgroups, 32 Xe-cores, and max 1024 work group size.

use std::fmt;

// ---------------------------------------------------------------------------
// Work dimension
// ---------------------------------------------------------------------------

/// Number of dimensions for an OpenCL dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkDimension {
    One,
    Two,
    Three,
}

impl WorkDimension {
    /// Number of dimensions as a `usize`.
    pub fn ndim(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

impl fmt::Display for WorkDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}D", self.ndim())
    }
}

// ---------------------------------------------------------------------------
// Work size configuration
// ---------------------------------------------------------------------------

/// Hardware-level configuration used by the optimizer.
#[derive(Debug, Clone)]
pub struct WorkSizeConfig {
    /// SIMD lane width (subgroup size). 16 for Intel Arc.
    pub simd_width: usize,
    /// Maximum threads in a single work group.
    pub max_work_group_size: usize,
    /// Preferred work group size for general kernels.
    pub preferred_work_group_size: usize,
    /// Number of compute units (Xe-cores on Arc).
    pub compute_units: usize,
}

impl Default for WorkSizeConfig {
    fn default() -> Self {
        IntelArcWorkSizeHints::default().into_config()
    }
}

// ---------------------------------------------------------------------------
// Intel Arc hardware hints
// ---------------------------------------------------------------------------

/// Hardware-specific hints for the Intel Arc family (e.g. A770).
#[derive(Debug, Clone)]
pub struct IntelArcWorkSizeHints {
    /// Subgroup (SIMD) width — 16 for Xe-HPG.
    pub simd_width: usize,
    /// Number of Xe-cores.
    pub xe_cores: usize,
    /// Maximum work group size supported by the device.
    pub max_work_group_size: usize,
    /// Preferred local work group size (multiple of `simd_width`).
    pub preferred_work_group_size: usize,
}

impl Default for IntelArcWorkSizeHints {
    fn default() -> Self {
        Self {
            simd_width: 16,
            xe_cores: 32,
            max_work_group_size: 1024,
            preferred_work_group_size: 256,
        }
    }
}

impl IntelArcWorkSizeHints {
    /// Convert hints into a [`WorkSizeConfig`].
    pub fn into_config(self) -> WorkSizeConfig {
        WorkSizeConfig {
            simd_width: self.simd_width,
            max_work_group_size: self.max_work_group_size,
            preferred_work_group_size: self.preferred_work_group_size,
            compute_units: self.xe_cores,
        }
    }
}

// ---------------------------------------------------------------------------
// Work size result
// ---------------------------------------------------------------------------

/// Result of a work size optimization.
#[derive(Debug, Clone)]
pub struct WorkSizeResult {
    /// Global work size per dimension.
    pub global_work_size: Vec<usize>,
    /// Local work size per dimension (`None` lets the runtime choose).
    pub local_work_size: Option<Vec<usize>>,
    /// Total number of work items across all dimensions.
    pub total_work_items: usize,
    /// Total number of work groups across all dimensions.
    pub total_work_groups: usize,
    /// Ratio of *useful* work items to total dispatched work items (0.0, 1.0].
    pub efficiency: f64,
}

impl WorkSizeResult {
    /// Dimension count derived from `global_work_size`.
    pub fn dimension(&self) -> WorkDimension {
        match self.global_work_size.len() {
            1 => WorkDimension::One,
            2 => WorkDimension::Two,
            _ => WorkDimension::Three,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `multiple`.
#[inline]
fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    let remainder = value % multiple;
    if remainder == 0 { value } else { value + multiple - remainder }
}

/// Clamp `local` so it does not exceed `max_work_group_size` while remaining
/// a multiple of `simd_width`.
#[inline]
fn clamp_local(local: usize, simd_width: usize, max_work_group_size: usize) -> usize {
    if local <= max_work_group_size {
        return local;
    }
    // Largest multiple of simd_width that fits.
    (max_work_group_size / simd_width) * simd_width
}

// ---------------------------------------------------------------------------
// WorkSizeOptimizer
// ---------------------------------------------------------------------------

/// Calculates optimal global/local work sizes for OpenCL dispatches.
#[derive(Debug, Clone, Default)]
pub struct WorkSizeOptimizer {
    config: WorkSizeConfig,
}

impl WorkSizeOptimizer {
    /// Create an optimizer with the given configuration.
    pub fn new(config: WorkSizeConfig) -> Self {
        Self { config }
    }

    /// Create an optimizer pre-configured for Intel Arc A770.
    pub fn intel_arc() -> Self {
        Self { config: IntelArcWorkSizeHints::default().into_config() }
    }

    /// Access the underlying config.
    pub fn config(&self) -> &WorkSizeConfig {
        &self.config
    }

    // ----- 1-D ----------------------------------------------------------

    /// Calculate optimal work sizes for 1-D elementwise operations.
    pub fn optimize_1d(&self, elements: usize) -> WorkSizeResult {
        let elements = elements.max(1);
        let local = clamp_local(
            self.config.preferred_work_group_size,
            self.config.simd_width,
            self.config.max_work_group_size,
        );
        let global = round_up(elements, local);
        let groups = global / local;
        let efficiency = elements as f64 / global as f64;

        WorkSizeResult {
            global_work_size: vec![global],
            local_work_size: Some(vec![local]),
            total_work_items: global,
            total_work_groups: groups,
            efficiency,
        }
    }

    // ----- 2-D ----------------------------------------------------------

    /// Calculate optimal work sizes for 2-D matrix operations.
    pub fn optimize_2d(&self, rows: usize, cols: usize) -> WorkSizeResult {
        let rows = rows.max(1);
        let cols = cols.max(1);

        let local_x = self.config.simd_width; // columns
        let local_y = (self.config.preferred_work_group_size / local_x)
            .min(self.config.max_work_group_size / local_x)
            .max(1);

        let global_x = round_up(cols, local_x);
        let global_y = round_up(rows, local_y);

        let total_items = global_x * global_y;
        let groups_x = global_x / local_x;
        let groups_y = global_y / local_y;
        let total_groups = groups_x * groups_y;
        let useful = rows * cols;
        let efficiency = useful as f64 / total_items as f64;

        WorkSizeResult {
            global_work_size: vec![global_x, global_y],
            local_work_size: Some(vec![local_x, local_y]),
            total_work_items: total_items,
            total_work_groups: total_groups,
            efficiency,
        }
    }

    // ----- 3-D ----------------------------------------------------------

    /// Calculate optimal work sizes for 3-D batched operations.
    pub fn optimize_3d(&self, batch: usize, rows: usize, cols: usize) -> WorkSizeResult {
        let batch = batch.max(1);
        let rows = rows.max(1);
        let cols = cols.max(1);

        let local_x = self.config.simd_width;
        // Budget for Y×Z combined must not exceed max_work_group_size / local_x.
        let yz_budget = (self.config.max_work_group_size / local_x).max(1);
        let local_y = yz_budget.clamp(1, rows);
        let local_z = (yz_budget / local_y).clamp(1, batch);

        let global_x = round_up(cols, local_x);
        let global_y = round_up(rows, local_y);
        let global_z = round_up(batch, local_z);

        let total_items = global_x * global_y * global_z;
        let total_groups = (global_x / local_x) * (global_y / local_y) * (global_z / local_z);
        let useful = batch * rows * cols;
        let efficiency = useful as f64 / total_items as f64;

        WorkSizeResult {
            global_work_size: vec![global_x, global_y, global_z],
            local_work_size: Some(vec![local_x, local_y, local_z]),
            total_work_items: total_items,
            total_work_groups: total_groups,
            efficiency,
        }
    }

    // ----- tiled matmul -------------------------------------------------

    /// Calculate optimal work sizes for tiled matrix multiplication.
    ///
    /// Each work group computes a `tile_size × tile_size` output tile.
    pub fn optimize_tiled_matmul(&self, m: usize, n: usize, tile_size: usize) -> WorkSizeResult {
        let m = m.max(1);
        let n = n.max(1);
        let tile_size = tile_size.max(1);

        let threads_per_group = tile_size * tile_size;
        let effective_tile = if threads_per_group > self.config.max_work_group_size {
            // Fall back to the largest square tile that fits.
            let side = (self.config.max_work_group_size as f64).sqrt() as usize;
            // Align down to simd_width.
            (side / self.config.simd_width) * self.config.simd_width
        } else {
            tile_size
        }
        .max(1);

        let local_x = effective_tile;
        let local_y = effective_tile;

        let global_x = round_up(n, local_x);
        let global_y = round_up(m, local_y);

        let total_items = global_x * global_y;
        let total_groups = (global_x / local_x) * (global_y / local_y);
        let useful = m * n;
        let efficiency = useful as f64 / total_items as f64;

        WorkSizeResult {
            global_work_size: vec![global_x, global_y],
            local_work_size: Some(vec![local_x, local_y]),
            total_work_items: total_items,
            total_work_groups: total_groups,
            efficiency,
        }
    }

    // ----- reduction ----------------------------------------------------

    /// Calculate optimal work sizes for row-wise reduction (softmax, norms).
    ///
    /// Each work group reduces a single row.
    pub fn optimize_reduction(&self, rows: usize, cols: usize) -> WorkSizeResult {
        let rows = rows.max(1);
        let cols = cols.max(1);

        let local = clamp_local(
            self.config.preferred_work_group_size,
            self.config.simd_width,
            self.config.max_work_group_size,
        );
        // One group per row; each group spans the row in strides of `local`.
        let global_x = local; // threads per row
        let global_y = rows; // one row per group-row

        let total_items = global_x * global_y;
        let total_groups = rows; // exactly one group per row
        let useful = rows * cols;
        let efficiency = useful as f64 / (total_items.max(1) * cols.div_ceil(local).max(1)) as f64;
        // Clamp to (0, 1].
        let efficiency = efficiency.clamp(f64::MIN_POSITIVE, 1.0);

        WorkSizeResult {
            global_work_size: vec![global_x, global_y],
            local_work_size: Some(vec![local, 1]),
            total_work_items: total_items,
            total_work_groups: total_groups,
            efficiency,
        }
    }
}

// ---------------------------------------------------------------------------
// Specialized helpers
// ---------------------------------------------------------------------------

/// Convenience wrapper for matrix-multiply work sizes.
pub struct MatmulWorkSize;

impl MatmulWorkSize {
    /// Optimal dispatch for an `m × n` output (using the default tile of 16).
    pub fn optimize(m: usize, n: usize) -> WorkSizeResult {
        WorkSizeOptimizer::intel_arc().optimize_tiled_matmul(m, n, 16)
    }

    /// Optimal dispatch with a custom tile size.
    pub fn optimize_with_tile(m: usize, n: usize, tile_size: usize) -> WorkSizeResult {
        WorkSizeOptimizer::intel_arc().optimize_tiled_matmul(m, n, tile_size)
    }
}

/// Convenience wrapper for reduction work sizes.
pub struct ReductionWorkSize;

impl ReductionWorkSize {
    /// Optimal dispatch for row-wise reduction over `rows × cols`.
    pub fn optimize(rows: usize, cols: usize) -> WorkSizeResult {
        WorkSizeOptimizer::intel_arc().optimize_reduction(rows, cols)
    }
}

/// Convenience wrapper for elementwise work sizes.
pub struct ElementwiseWorkSize;

impl ElementwiseWorkSize {
    /// Optimal dispatch for `n` independent elements.
    pub fn optimize(n: usize) -> WorkSizeResult {
        WorkSizeOptimizer::intel_arc().optimize_1d(n)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn arc_optimizer() -> WorkSizeOptimizer {
        WorkSizeOptimizer::intel_arc()
    }

    /// Assert invariants that must hold for every result.
    fn assert_result_invariants(r: &WorkSizeResult, required_elements: usize) {
        // Global >= required
        let total_global: usize = r.global_work_size.iter().product();
        assert!(
            total_global >= required_elements,
            "global {total_global} < required {required_elements}"
        );
        assert_common_invariants(r);
    }

    /// Invariants that apply to all results (including reduction where
    /// the global work size intentionally does not cover every element).
    fn assert_common_invariants(r: &WorkSizeResult) {
        // Efficiency in (0, 1]
        assert!(r.efficiency > 0.0, "efficiency must be > 0, got {}", r.efficiency);
        assert!(r.efficiency <= 1.0, "efficiency must be <= 1.0, got {}", r.efficiency);
        // Work groups > 0
        assert!(r.total_work_groups > 0);
        // Local sizes (if present) divide global evenly
        if let Some(ref local) = r.local_work_size {
            for (g, l) in r.global_work_size.iter().zip(local.iter()) {
                assert_eq!(g % l, 0, "global {g} not divisible by local {l}");
            }
            // Local product <= max work group size (1024 for Arc)
            let local_product: usize = local.iter().product();
            assert!(local_product <= 1024, "local product {local_product} > 1024");
        }
    }

    // -----------------------------------------------------------------------
    // IntelArcWorkSizeHints
    // -----------------------------------------------------------------------

    #[test]
    fn intel_arc_defaults() {
        let hints = IntelArcWorkSizeHints::default();
        assert_eq!(hints.simd_width, 16);
        assert_eq!(hints.xe_cores, 32);
        assert_eq!(hints.max_work_group_size, 1024);
        assert_eq!(hints.preferred_work_group_size, 256);
    }

    #[test]
    fn intel_arc_config_conversion() {
        let cfg = IntelArcWorkSizeHints::default().into_config();
        assert_eq!(cfg.simd_width, 16);
        assert_eq!(cfg.compute_units, 32);
        assert_eq!(cfg.max_work_group_size, 1024);
        assert_eq!(cfg.preferred_work_group_size, 256);
    }

    #[test]
    fn work_size_config_default_matches_arc() {
        let cfg = WorkSizeConfig::default();
        assert_eq!(cfg.simd_width, 16);
        assert_eq!(cfg.compute_units, 32);
    }

    // -----------------------------------------------------------------------
    // WorkDimension
    // -----------------------------------------------------------------------

    #[test]
    fn work_dimension_ndim() {
        assert_eq!(WorkDimension::One.ndim(), 1);
        assert_eq!(WorkDimension::Two.ndim(), 2);
        assert_eq!(WorkDimension::Three.ndim(), 3);
    }

    #[test]
    fn work_dimension_display() {
        assert_eq!(format!("{}", WorkDimension::One), "1D");
        assert_eq!(format!("{}", WorkDimension::Two), "2D");
        assert_eq!(format!("{}", WorkDimension::Three), "3D");
    }

    // -----------------------------------------------------------------------
    // round_up helper
    // -----------------------------------------------------------------------

    #[test]
    fn round_up_exact_multiple() {
        assert_eq!(round_up(256, 256), 256);
        assert_eq!(round_up(512, 256), 512);
    }

    #[test]
    fn round_up_non_multiple() {
        assert_eq!(round_up(1, 256), 256);
        assert_eq!(round_up(257, 256), 512);
        assert_eq!(round_up(1000, 256), 1024);
    }

    #[test]
    fn round_up_zero_multiple() {
        assert_eq!(round_up(42, 0), 42);
    }

    // -----------------------------------------------------------------------
    // 1-D elementwise
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_1d_single_element() {
        let r = arc_optimizer().optimize_1d(1);
        assert_result_invariants(&r, 1);
        assert_eq!(r.global_work_size.len(), 1);
        assert_eq!(r.dimension(), WorkDimension::One);
    }

    #[test]
    fn optimize_1d_medium() {
        let r = arc_optimizer().optimize_1d(1000);
        assert_result_invariants(&r, 1000);
        assert!(r.global_work_size[0] >= 1000);
        assert_eq!(r.global_work_size[0] % 256, 0);
    }

    #[test]
    fn optimize_1d_large() {
        let r = arc_optimizer().optimize_1d(1_000_000);
        assert_result_invariants(&r, 1_000_000);
        assert!(r.global_work_size[0] >= 1_000_000);
    }

    #[test]
    fn optimize_1d_exact_multiple_of_local() {
        let r = arc_optimizer().optimize_1d(256);
        assert_result_invariants(&r, 256);
        assert_eq!(r.global_work_size[0], 256);
        assert!((r.efficiency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn optimize_1d_exact_multiple_512() {
        let r = arc_optimizer().optimize_1d(512);
        assert_result_invariants(&r, 512);
        assert_eq!(r.global_work_size[0], 512);
        assert!((r.efficiency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn optimize_1d_zero_elements_clamped() {
        let r = arc_optimizer().optimize_1d(0);
        assert_result_invariants(&r, 1);
    }

    #[test]
    fn optimize_1d_power_of_two() {
        let r = arc_optimizer().optimize_1d(65536);
        assert_result_invariants(&r, 65536);
        assert_eq!(r.global_work_size[0], 65536);
    }

    #[test]
    fn optimize_1d_local_size_multiple_of_simd() {
        let r = arc_optimizer().optimize_1d(500);
        let local = r.local_work_size.as_ref().unwrap()[0];
        assert_eq!(local % 16, 0, "local {local} not multiple of SIMD width 16");
    }

    // -----------------------------------------------------------------------
    // 2-D matrix
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_2d_square_small() {
        let r = arc_optimizer().optimize_2d(16, 16);
        assert_result_invariants(&r, 16 * 16);
        assert_eq!(r.global_work_size.len(), 2);
    }

    #[test]
    fn optimize_2d_square_large() {
        let r = arc_optimizer().optimize_2d(1024, 1024);
        assert_result_invariants(&r, 1024 * 1024);
    }

    #[test]
    fn optimize_2d_rectangular_wide() {
        let r = arc_optimizer().optimize_2d(4, 4096);
        assert_result_invariants(&r, 4 * 4096);
        assert!(r.global_work_size[0] >= 4096);
        assert!(r.global_work_size[1] >= 4);
    }

    #[test]
    fn optimize_2d_rectangular_tall() {
        let r = arc_optimizer().optimize_2d(4096, 4);
        assert_result_invariants(&r, 4096 * 4);
        assert!(r.global_work_size[0] >= 4);
        assert!(r.global_work_size[1] >= 4096);
    }

    #[test]
    fn optimize_2d_non_power_of_2() {
        let r = arc_optimizer().optimize_2d(100, 300);
        assert_result_invariants(&r, 100 * 300);
    }

    #[test]
    fn optimize_2d_single_row() {
        let r = arc_optimizer().optimize_2d(1, 512);
        assert_result_invariants(&r, 512);
    }

    #[test]
    fn optimize_2d_single_col() {
        let r = arc_optimizer().optimize_2d(512, 1);
        assert_result_invariants(&r, 512);
    }

    #[test]
    fn optimize_2d_dimension_enum() {
        let r = arc_optimizer().optimize_2d(32, 32);
        assert_eq!(r.dimension(), WorkDimension::Two);
    }

    // -----------------------------------------------------------------------
    // 3-D batched
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_3d_basic() {
        let r = arc_optimizer().optimize_3d(4, 64, 64);
        assert_result_invariants(&r, 4 * 64 * 64);
        assert_eq!(r.global_work_size.len(), 3);
        assert_eq!(r.dimension(), WorkDimension::Three);
    }

    #[test]
    fn optimize_3d_single_batch() {
        let r = arc_optimizer().optimize_3d(1, 128, 128);
        assert_result_invariants(&r, 128 * 128);
        assert_eq!(r.global_work_size[2], 1); // batch dim
    }

    #[test]
    fn optimize_3d_large_batch() {
        let r = arc_optimizer().optimize_3d(32, 64, 64);
        assert_result_invariants(&r, 32 * 64 * 64);
    }

    #[test]
    fn optimize_3d_non_power_of_2() {
        let r = arc_optimizer().optimize_3d(3, 100, 200);
        assert_result_invariants(&r, 3 * 100 * 200);
    }

    #[test]
    fn optimize_3d_tiny() {
        let r = arc_optimizer().optimize_3d(1, 1, 1);
        assert_result_invariants(&r, 1);
    }

    #[test]
    fn optimize_3d_zero_clamped() {
        let r = arc_optimizer().optimize_3d(0, 0, 0);
        assert_result_invariants(&r, 1);
    }

    // -----------------------------------------------------------------------
    // Tiled matmul
    // -----------------------------------------------------------------------

    #[test]
    fn tiled_matmul_perfect_tiles() {
        let r = arc_optimizer().optimize_tiled_matmul(64, 64, 16);
        assert_result_invariants(&r, 64 * 64);
        assert_eq!(r.global_work_size[0], 64);
        assert_eq!(r.global_work_size[1], 64);
        assert!((r.efficiency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tiled_matmul_remainder_tiles() {
        let r = arc_optimizer().optimize_tiled_matmul(100, 100, 16);
        assert_result_invariants(&r, 100 * 100);
        assert!(r.global_work_size[0] >= 100);
        assert!(r.global_work_size[1] >= 100);
        assert!(r.efficiency < 1.0);
    }

    #[test]
    fn tiled_matmul_small_matrix() {
        let r = arc_optimizer().optimize_tiled_matmul(4, 4, 16);
        assert_result_invariants(&r, 4 * 4);
        assert!(r.global_work_size[0] >= 4);
        assert!(r.global_work_size[1] >= 4);
    }

    #[test]
    fn tiled_matmul_large_tile_clamped() {
        // 64×64 = 4096 > 1024, should fall back.
        let r = arc_optimizer().optimize_tiled_matmul(256, 256, 64);
        assert_result_invariants(&r, 256 * 256);
        let local = r.local_work_size.as_ref().unwrap();
        assert!(local[0] * local[1] <= 1024);
    }

    #[test]
    fn tiled_matmul_tile_1() {
        let r = arc_optimizer().optimize_tiled_matmul(32, 32, 1);
        assert_result_invariants(&r, 32 * 32);
    }

    #[test]
    fn tiled_matmul_non_square() {
        let r = arc_optimizer().optimize_tiled_matmul(128, 256, 16);
        assert_result_invariants(&r, 128 * 256);
        assert!(r.global_work_size[0] >= 256);
        assert!(r.global_work_size[1] >= 128);
    }

    #[test]
    fn tiled_matmul_convenience() {
        let r = MatmulWorkSize::optimize(64, 64);
        assert_result_invariants(&r, 64 * 64);
    }

    #[test]
    fn tiled_matmul_convenience_custom_tile() {
        let r = MatmulWorkSize::optimize_with_tile(128, 128, 32);
        assert_result_invariants(&r, 128 * 128);
    }

    // -----------------------------------------------------------------------
    // Reduction
    // -----------------------------------------------------------------------

    #[test]
    fn reduction_single_row() {
        let r = arc_optimizer().optimize_reduction(1, 1024);
        assert_common_invariants(&r);
        assert_eq!(r.total_work_groups, 1);
        assert!(r.global_work_size[1] >= 1); // rows dimension
    }

    #[test]
    fn reduction_multiple_rows() {
        let r = arc_optimizer().optimize_reduction(128, 512);
        assert_common_invariants(&r);
        assert_eq!(r.total_work_groups, 128);
        assert!(r.global_work_size[1] >= 128);
    }

    #[test]
    fn reduction_large_vocab() {
        let r = arc_optimizer().optimize_reduction(1, 128256);
        assert_common_invariants(&r);
        assert_eq!(r.total_work_groups, 1);
    }

    #[test]
    fn reduction_small_cols() {
        let r = arc_optimizer().optimize_reduction(64, 8);
        assert_common_invariants(&r);
        assert_eq!(r.total_work_groups, 64);
    }

    #[test]
    fn reduction_convenience() {
        let r = ReductionWorkSize::optimize(32, 256);
        assert_common_invariants(&r);
        assert_eq!(r.total_work_groups, 32);
    }

    // -----------------------------------------------------------------------
    // ElementwiseWorkSize convenience
    // -----------------------------------------------------------------------

    #[test]
    fn elementwise_convenience() {
        let r = ElementwiseWorkSize::optimize(10000);
        assert_result_invariants(&r, 10000);
        assert_eq!(r.dimension(), WorkDimension::One);
    }

    #[test]
    fn elementwise_convenience_small() {
        let r = ElementwiseWorkSize::optimize(1);
        assert_result_invariants(&r, 1);
    }

    // -----------------------------------------------------------------------
    // Efficiency bounds
    // -----------------------------------------------------------------------

    #[test]
    fn efficiency_always_positive() {
        for n in [1, 2, 7, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025] {
            let r = arc_optimizer().optimize_1d(n);
            assert!(r.efficiency > 0.0, "n={n}: efficiency={}", r.efficiency);
            assert!(r.efficiency <= 1.0, "n={n}: efficiency={}", r.efficiency);
        }
    }

    #[test]
    fn efficiency_2d_bounds() {
        for (rows, cols) in [(1, 1), (7, 13), (16, 16), (100, 200), (1024, 1024)] {
            let r = arc_optimizer().optimize_2d(rows, cols);
            assert!(
                r.efficiency > 0.0 && r.efficiency <= 1.0,
                "rows={rows}, cols={cols}: eff={}",
                r.efficiency
            );
        }
    }

    #[test]
    fn efficiency_3d_bounds() {
        for (b, r, c) in [(1, 1, 1), (2, 32, 32), (8, 64, 128)] {
            let res = arc_optimizer().optimize_3d(b, r, c);
            assert!(
                res.efficiency > 0.0 && res.efficiency <= 1.0,
                "b={b}, r={r}, c={c}: eff={}",
                res.efficiency
            );
        }
    }

    #[test]
    fn efficiency_matmul_bounds() {
        for (m, n) in [(1, 1), (16, 16), (100, 100), (1024, 1024)] {
            let r = arc_optimizer().optimize_tiled_matmul(m, n, 16);
            assert!(
                r.efficiency > 0.0 && r.efficiency <= 1.0,
                "m={m}, n={n}: eff={}",
                r.efficiency
            );
        }
    }

    #[test]
    fn efficiency_reduction_bounds() {
        for (rows, cols) in [(1, 1), (1, 128256), (64, 512)] {
            let r = arc_optimizer().optimize_reduction(rows, cols);
            assert!(
                r.efficiency > 0.0 && r.efficiency <= 1.0,
                "rows={rows}, cols={cols}: eff={}",
                r.efficiency
            );
        }
    }

    // -----------------------------------------------------------------------
    // Global size >= required
    // -----------------------------------------------------------------------

    #[test]
    fn global_ge_required_1d() {
        for n in [1, 100, 1000, 1_000_000] {
            let r = arc_optimizer().optimize_1d(n);
            assert!(r.global_work_size[0] >= n);
        }
    }

    #[test]
    fn global_ge_required_2d() {
        let r = arc_optimizer().optimize_2d(100, 200);
        assert!(r.global_work_size[0] >= 200);
        assert!(r.global_work_size[1] >= 100);
    }

    #[test]
    fn global_ge_required_tiled() {
        let r = arc_optimizer().optimize_tiled_matmul(100, 200, 16);
        assert!(r.global_work_size[0] >= 200);
        assert!(r.global_work_size[1] >= 100);
    }

    // -----------------------------------------------------------------------
    // Work group size never exceeds max
    // -----------------------------------------------------------------------

    #[test]
    fn local_never_exceeds_max_1d() {
        let r = arc_optimizer().optimize_1d(1_000_000);
        let local: usize = r.local_work_size.as_ref().unwrap().iter().product();
        assert!(local <= 1024);
    }

    #[test]
    fn local_never_exceeds_max_2d() {
        let r = arc_optimizer().optimize_2d(4096, 4096);
        let local: usize = r.local_work_size.as_ref().unwrap().iter().product();
        assert!(local <= 1024);
    }

    #[test]
    fn local_never_exceeds_max_3d() {
        let r = arc_optimizer().optimize_3d(32, 128, 128);
        let local: usize = r.local_work_size.as_ref().unwrap().iter().product();
        assert!(local <= 1024);
    }

    #[test]
    fn local_never_exceeds_max_matmul() {
        for tile in [8, 16, 32, 64, 128] {
            let r = arc_optimizer().optimize_tiled_matmul(512, 512, tile);
            let local: usize = r.local_work_size.as_ref().unwrap().iter().product();
            assert!(local <= 1024, "tile={tile}: local product={local}");
        }
    }

    // -----------------------------------------------------------------------
    // Custom config
    // -----------------------------------------------------------------------

    #[test]
    fn custom_config_small_simd() {
        let cfg = WorkSizeConfig {
            simd_width: 8,
            max_work_group_size: 512,
            preferred_work_group_size: 128,
            compute_units: 16,
        };
        let opt = WorkSizeOptimizer::new(cfg);
        let r = opt.optimize_1d(1000);
        assert_result_invariants(&r, 1000);
        let local = r.local_work_size.as_ref().unwrap()[0];
        assert_eq!(local, 128);
    }

    #[test]
    fn custom_config_large_simd() {
        let cfg = WorkSizeConfig {
            simd_width: 32,
            max_work_group_size: 2048,
            preferred_work_group_size: 512,
            compute_units: 64,
        };
        let opt = WorkSizeOptimizer::new(cfg);
        let r = opt.optimize_1d(10000);
        assert_result_invariants(&r, 10000);
        let local = r.local_work_size.as_ref().unwrap()[0];
        assert_eq!(local % 32, 0);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_1d_max_usize_like() {
        // Large but not truly usize::MAX (would OOM). Test a reasonably large value.
        let r = arc_optimizer().optimize_1d(1 << 24);
        assert_result_invariants(&r, 1 << 24);
    }

    #[test]
    fn optimize_2d_one_by_one() {
        let r = arc_optimizer().optimize_2d(1, 1);
        assert_result_invariants(&r, 1);
    }

    #[test]
    fn optimize_3d_large_cols_small_rest() {
        let r = arc_optimizer().optimize_3d(1, 1, 100_000);
        assert_result_invariants(&r, 100_000);
    }

    #[test]
    fn tiled_matmul_zero_clamped() {
        let r = arc_optimizer().optimize_tiled_matmul(0, 0, 0);
        assert_result_invariants(&r, 1);
    }

    #[test]
    fn reduction_zero_clamped() {
        let r = arc_optimizer().optimize_reduction(0, 0);
        assert_common_invariants(&r);
    }
}
