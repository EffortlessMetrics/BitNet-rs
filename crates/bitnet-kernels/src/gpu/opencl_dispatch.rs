//! A770-specific OpenCL dispatch sizing with workgroup constraint handling.
//!
//! Provides dispatch configuration for Intel Arc A770 (Xe-HPG) architecture,
//! including per-dimension workgroup limits, subgroup size selection, and
//! local memory budget validation. Complements [`crate::opencl_work_size`]
//! with stricter, device-specific constraint checking.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from dispatch validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// A local work-size dimension exceeds the per-axis device limit.
    LocalDimExceedsLimit { dim: usize, local: usize, limit: usize },
    /// Total local work-group size exceeds device maximum.
    WorkgroupSizeExceeded { total: usize, max: usize },
    /// Global work size is not a multiple of local work size in some dimension.
    GlobalNotMultipleOfLocal { dim: usize, global: usize, local: usize },
    /// Dispatch dimensionality must be 1, 2, or 3.
    InvalidDimensionCount { count: usize },
    /// Global and local arrays have different lengths.
    DimensionMismatch { global_dims: usize, local_dims: usize },
    /// Requested local memory exceeds the device SLM budget.
    LocalMemoryExceeded { requested: usize, available: usize },
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalDimExceedsLimit { dim, local, limit } => {
                write!(f, "local work-size dim {dim} is {local}, exceeds limit {limit}")
            }
            Self::WorkgroupSizeExceeded { total, max } => {
                write!(f, "workgroup size {total} exceeds max {max}")
            }
            Self::GlobalNotMultipleOfLocal { dim, global, local } => {
                write!(f, "global[{dim}]={global} not a multiple of local[{dim}]={local}")
            }
            Self::InvalidDimensionCount { count } => {
                write!(f, "invalid dimension count {count} (expected 1..=3)")
            }
            Self::DimensionMismatch { global_dims, local_dims } => {
                write!(f, "dimension mismatch: global has {global_dims}, local has {local_dims}")
            }
            Self::LocalMemoryExceeded { requested, available } => {
                write!(f, "local memory {requested} bytes exceeds {available} bytes available")
            }
        }
    }
}

impl std::error::Error for DispatchError {}

// ---------------------------------------------------------------------------
// A770 workgroup limits
// ---------------------------------------------------------------------------

/// Hardware limits for the Intel Arc A770 (Xe-HPG) GPU.
///
/// These constants are derived from the OpenCL device queries
/// (`CL_DEVICE_MAX_WORK_GROUP_SIZE`, `CL_DEVICE_MAX_WORK_ITEM_SIZES`, etc.)
/// on the A770 with Intel's compute runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct A770WorkgroupLimits {
    /// Maximum total work-items in a single workgroup (product of all dims).
    pub max_workgroup_size: usize,
    /// Per-dimension maximum work-group sizes `[x, y, z]`.
    pub max_workgroup_dims: [usize; 3],
    /// Preferred workgroup size multiple (EU thread scheduling granularity).
    pub preferred_workgroup_multiple: usize,
    /// Number of compute units (Xe-cores).
    pub max_compute_units: usize,
    /// SIMD lane width for Xe-HPG (typically 16; some kernels use 32).
    pub subgroup_size: usize,
    /// Shared Local Memory per sub-slice in bytes (64 KB).
    pub local_memory_size: usize,
}

impl Default for A770WorkgroupLimits {
    fn default() -> Self {
        Self {
            max_workgroup_size: 1024,
            max_workgroup_dims: [1024, 1024, 64],
            preferred_workgroup_multiple: 32,
            max_compute_units: 32,
            subgroup_size: 16,
            local_memory_size: 65536,
        }
    }
}

impl A770WorkgroupLimits {
    /// Create limits with a specific subgroup size (16 or 32).
    pub fn with_subgroup_size(mut self, size: usize) -> Self {
        self.subgroup_size = size;
        self
    }
}

// ---------------------------------------------------------------------------
// Dispatch config
// ---------------------------------------------------------------------------

/// A validated dispatch configuration ready to be passed to
/// `clEnqueueNDRangeKernel`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchConfig {
    /// Global work size per dimension.
    pub global_work_size: Vec<usize>,
    /// Local work size per dimension.
    pub local_work_size: Vec<usize>,
}

impl DispatchConfig {
    /// Build a dispatch config, auto-rounding globals to the nearest multiple
    /// of the corresponding local dimension and validating against `limits`.
    pub fn new(
        global: &[usize],
        local: &[usize],
        limits: &A770WorkgroupLimits,
    ) -> Result<Self, DispatchError> {
        if global.len() != local.len() {
            return Err(DispatchError::DimensionMismatch {
                global_dims: global.len(),
                local_dims: local.len(),
            });
        }

        // Auto-round each global dimension up to a multiple of local.
        let rounded_global: Vec<usize> =
            global.iter().zip(local.iter()).map(|(&g, &l)| round_up(g.max(1), l.max(1))).collect();

        validate_dispatch(&rounded_global, local, limits)?;

        Ok(Self { global_work_size: rounded_global, local_work_size: local.to_vec() })
    }

    /// Number of dispatch dimensions.
    pub fn ndim(&self) -> usize {
        self.global_work_size.len()
    }

    /// Total number of work-groups that will be launched.
    pub fn work_group_count(&self) -> usize {
        self.global_work_size.iter().zip(self.local_work_size.iter()).map(|(g, l)| g / l).product()
    }
}

impl fmt::Display for DispatchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "global={:?} local={:?} groups={}",
            self.global_work_size,
            self.local_work_size,
            self.work_group_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Public dispatch helpers
// ---------------------------------------------------------------------------

/// Compute optimal 1-D dispatch sizes for `n` elements.
///
/// Returns `(global_work_size, local_work_size)`. The local size is the
/// preferred workgroup multiple clamped to the device limit.
pub fn dispatch_1d(n: usize, limits: &A770WorkgroupLimits) -> (usize, usize) {
    let n = n.max(1);
    let local = limits
        .preferred_workgroup_multiple
        .min(limits.max_workgroup_size)
        .min(limits.max_workgroup_dims[0])
        .max(1);
    let global = round_up(n, local);
    (global, local)
}

/// Compute optimal 2-D dispatch sizes for an `m × n` output (e.g. matmul).
///
/// Returns `(global[2], local[2])` where `global[0]` covers columns (`n`)
/// and `global[1]` covers rows (`m`), following the OpenCL convention of
/// fastest-varying dimension first.
pub fn dispatch_2d(m: usize, n: usize, limits: &A770WorkgroupLimits) -> ([usize; 2], [usize; 2]) {
    let m = m.max(1);
    let n = n.max(1);

    // local_x = subgroup_size (SIMD-width-aligned columns)
    let local_x = limits.subgroup_size.min(limits.max_workgroup_dims[0]).max(1);

    // local_y fills remaining budget (rows per workgroup).
    let max_local_y =
        (limits.max_workgroup_size / local_x).min(limits.max_workgroup_dims[1]).max(1);
    // Pick the largest power-of-2 ≤ max_local_y for efficient tiling.
    let local_y = prev_power_of_two(max_local_y).max(1);

    let global_x = round_up(n, local_x);
    let global_y = round_up(m, local_y);

    ([global_x, global_y], [local_x, local_y])
}

/// Validate a dispatch configuration against A770 limits.
///
/// Checks per-dimension bounds, total workgroup size, and global/local
/// divisibility.
pub fn validate_dispatch(
    global: &[usize],
    local: &[usize],
    limits: &A770WorkgroupLimits,
) -> Result<(), DispatchError> {
    let ndim = global.len();
    if ndim == 0 || ndim > 3 {
        return Err(DispatchError::InvalidDimensionCount { count: ndim });
    }
    if local.len() != ndim {
        return Err(DispatchError::DimensionMismatch {
            global_dims: ndim,
            local_dims: local.len(),
        });
    }

    // Per-dimension checks.
    for (i, (&l, &g)) in local.iter().zip(global.iter()).enumerate() {
        if l > limits.max_workgroup_dims[i] {
            return Err(DispatchError::LocalDimExceedsLimit {
                dim: i,
                local: l,
                limit: limits.max_workgroup_dims[i],
            });
        }
        if g % l != 0 {
            return Err(DispatchError::GlobalNotMultipleOfLocal { dim: i, global: g, local: l });
        }
    }

    // Total workgroup size.
    let total: usize = local.iter().product();
    if total > limits.max_workgroup_size {
        return Err(DispatchError::WorkgroupSizeExceeded { total, max: limits.max_workgroup_size });
    }

    Ok(())
}

/// Validate that `bytes` of local (shared) memory fits within the A770 SLM.
pub fn validate_local_memory(
    bytes: usize,
    limits: &A770WorkgroupLimits,
) -> Result<(), DispatchError> {
    if bytes > limits.local_memory_size {
        return Err(DispatchError::LocalMemoryExceeded {
            requested: bytes,
            available: limits.local_memory_size,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `multiple`.
#[inline]
fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    let rem = value % multiple;
    if rem == 0 { value } else { value + multiple - rem }
}

/// Largest power of two ≤ `n`. Returns 1 when `n == 0`.
#[inline]
fn prev_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1 << (usize::BITS - 1 - n.leading_zeros())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn a770() -> A770WorkgroupLimits {
        A770WorkgroupLimits::default()
    }

    // -- A770WorkgroupLimits ------------------------------------------------

    #[test]
    fn default_limits_match_a770_spec() {
        let l = a770();
        assert_eq!(l.max_workgroup_size, 1024);
        assert_eq!(l.max_workgroup_dims, [1024, 1024, 64]);
        assert_eq!(l.preferred_workgroup_multiple, 32);
        assert_eq!(l.max_compute_units, 32);
        assert_eq!(l.subgroup_size, 16);
        assert_eq!(l.local_memory_size, 65536);
    }

    #[test]
    fn with_subgroup_size_32() {
        let l = A770WorkgroupLimits::default().with_subgroup_size(32);
        assert_eq!(l.subgroup_size, 32);
        // Other fields unchanged.
        assert_eq!(l.max_workgroup_size, 1024);
    }

    // -- round_up -----------------------------------------------------------

    #[test]
    fn round_up_exact() {
        assert_eq!(round_up(32, 32), 32);
        assert_eq!(round_up(1024, 16), 1024);
    }

    #[test]
    fn round_up_non_exact() {
        assert_eq!(round_up(1, 32), 32);
        assert_eq!(round_up(33, 32), 64);
        assert_eq!(round_up(1000, 16), 1008);
    }

    #[test]
    fn round_up_zero_multiple() {
        assert_eq!(round_up(42, 0), 42);
    }

    // -- prev_power_of_two --------------------------------------------------

    #[test]
    fn prev_pow2() {
        assert_eq!(prev_power_of_two(0), 1);
        assert_eq!(prev_power_of_two(1), 1);
        assert_eq!(prev_power_of_two(2), 2);
        assert_eq!(prev_power_of_two(3), 2);
        assert_eq!(prev_power_of_two(4), 4);
        assert_eq!(prev_power_of_two(5), 4);
        assert_eq!(prev_power_of_two(64), 64);
        assert_eq!(prev_power_of_two(65), 64);
        assert_eq!(prev_power_of_two(1024), 1024);
    }

    // -- dispatch_1d --------------------------------------------------------

    #[test]
    fn dispatch_1d_single_element() {
        let (g, l) = dispatch_1d(1, &a770());
        assert_eq!(l, 32); // preferred_workgroup_multiple
        assert_eq!(g, 32);
        assert_eq!(g % l, 0);
    }

    #[test]
    fn dispatch_1d_exact_multiple() {
        let (g, l) = dispatch_1d(1024, &a770());
        assert_eq!(g, 1024);
        assert_eq!(g % l, 0);
    }

    #[test]
    fn dispatch_1d_non_power_of_two() {
        let (g, l) = dispatch_1d(1000, &a770());
        assert!(g >= 1000);
        assert_eq!(g % l, 0);
    }

    #[test]
    fn dispatch_1d_very_large() {
        let (g, l) = dispatch_1d(10_000_000, &a770());
        assert!(g >= 10_000_000);
        assert_eq!(g % l, 0);
        assert!(l <= 1024);
    }

    #[test]
    fn dispatch_1d_zero_clamped() {
        let (g, l) = dispatch_1d(0, &a770());
        assert!(g >= 1);
        assert_eq!(g % l, 0);
    }

    // -- dispatch_2d --------------------------------------------------------

    #[test]
    fn dispatch_2d_square_512() {
        let (g, l) = dispatch_2d(512, 512, &a770());
        assert!(g[0] >= 512);
        assert!(g[1] >= 512);
        assert_eq!(g[0] % l[0], 0);
        assert_eq!(g[1] % l[1], 0);
        assert!(l[0] * l[1] <= 1024);
    }

    #[test]
    fn dispatch_2d_square_2048() {
        let (g, l) = dispatch_2d(2048, 2048, &a770());
        assert!(g[0] >= 2048);
        assert!(g[1] >= 2048);
        assert_eq!(g[0] % l[0], 0);
        assert_eq!(g[1] % l[1], 0);
        assert!(l[0] * l[1] <= 1024);
    }

    #[test]
    fn dispatch_2d_llm_ffn_shape() {
        // Common BitNet FFN shape: 4096 × 11008
        let (g, l) = dispatch_2d(4096, 11008, &a770());
        assert!(g[0] >= 11008);
        assert!(g[1] >= 4096);
        assert_eq!(g[0] % l[0], 0);
        assert_eq!(g[1] % l[1], 0);
        assert!(l[0] * l[1] <= 1024);
    }

    #[test]
    fn dispatch_2d_single_row() {
        let (g, l) = dispatch_2d(1, 4096, &a770());
        assert!(g[0] >= 4096);
        assert!(g[1] >= 1);
        assert_eq!(g[0] % l[0], 0);
        assert_eq!(g[1] % l[1], 0);
    }

    #[test]
    fn dispatch_2d_single_col() {
        let (g, l) = dispatch_2d(4096, 1, &a770());
        assert!(g[0] >= 1);
        assert!(g[1] >= 4096);
        assert_eq!(g[0] % l[0], 0);
        assert_eq!(g[1] % l[1], 0);
    }

    #[test]
    fn dispatch_2d_non_power_of_two() {
        let (g, l) = dispatch_2d(100, 300, &a770());
        assert!(g[0] >= 300);
        assert!(g[1] >= 100);
        assert_eq!(g[0] % l[0], 0);
        assert_eq!(g[1] % l[1], 0);
        assert!(l[0] * l[1] <= 1024);
    }

    #[test]
    fn dispatch_2d_tiny() {
        let (g, l) = dispatch_2d(1, 1, &a770());
        assert!(g[0] >= 1);
        assert!(g[1] >= 1);
        assert_eq!(g[0] % l[0], 0);
        assert_eq!(g[1] % l[1], 0);
    }

    #[test]
    fn dispatch_2d_subgroup_32() {
        let limits = A770WorkgroupLimits::default().with_subgroup_size(32);
        let (g, l) = dispatch_2d(512, 512, &limits);
        assert_eq!(l[0], 32, "local_x should be subgroup_size=32");
        assert!(g[0] >= 512);
        assert_eq!(g[0] % l[0], 0);
        assert!(l[0] * l[1] <= 1024);
    }

    // -- validate_dispatch --------------------------------------------------

    #[test]
    fn validate_ok_1d() {
        assert!(validate_dispatch(&[1024], &[32], &a770()).is_ok());
    }

    #[test]
    fn validate_ok_2d() {
        assert!(validate_dispatch(&[512, 512], &[16, 16], &a770()).is_ok());
    }

    #[test]
    fn validate_ok_3d() {
        assert!(validate_dispatch(&[256, 128, 4], &[16, 8, 4], &a770()).is_ok());
    }

    #[test]
    fn validate_err_dim_mismatch() {
        let err = validate_dispatch(&[256, 256], &[16], &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::DimensionMismatch { .. }));
    }

    #[test]
    fn validate_err_zero_dims() {
        let err = validate_dispatch(&[], &[], &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::InvalidDimensionCount { count: 0 }));
    }

    #[test]
    fn validate_err_four_dims() {
        let err = validate_dispatch(&[64, 64, 64, 64], &[8, 8, 8, 8], &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::InvalidDimensionCount { count: 4 }));
    }

    #[test]
    fn validate_err_local_dim_exceeds_z_limit() {
        // Z limit on A770 is 64.
        let err = validate_dispatch(&[128, 128, 128], &[16, 8, 128], &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::LocalDimExceedsLimit { dim: 2, limit: 64, .. }));
    }

    #[test]
    fn validate_err_workgroup_size_exceeded() {
        // 512 * 4 = 2048 > 1024.
        let err = validate_dispatch(&[512, 4], &[512, 4], &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::WorkgroupSizeExceeded { total: 2048, max: 1024 }));
    }

    #[test]
    fn validate_err_global_not_multiple() {
        let err = validate_dispatch(&[100], &[32], &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::GlobalNotMultipleOfLocal { dim: 0, .. }));
    }

    // -- validate_local_memory ----------------------------------------------

    #[test]
    fn local_memory_ok() {
        assert!(validate_local_memory(32768, &a770()).is_ok());
        assert!(validate_local_memory(65536, &a770()).is_ok());
    }

    #[test]
    fn local_memory_exceeded() {
        let err = validate_local_memory(65537, &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::LocalMemoryExceeded { .. }));
    }

    // -- DispatchConfig -----------------------------------------------------

    #[test]
    fn dispatch_config_auto_rounds() {
        let cfg = DispatchConfig::new(&[1000], &[32], &a770()).unwrap();
        assert_eq!(cfg.global_work_size[0], 1024);
        assert_eq!(cfg.local_work_size[0], 32);
        assert_eq!(cfg.work_group_count(), 32);
    }

    #[test]
    fn dispatch_config_2d() {
        let cfg = DispatchConfig::new(&[100, 200], &[16, 16], &a770()).unwrap();
        assert!(cfg.global_work_size[0] >= 100);
        assert!(cfg.global_work_size[1] >= 200);
        assert_eq!(cfg.ndim(), 2);
    }

    #[test]
    fn dispatch_config_rejects_bad_local() {
        // local_z = 128 exceeds dim-2 limit of 64.
        let err = DispatchConfig::new(&[64, 64, 128], &[16, 4, 128], &a770()).unwrap_err();
        assert!(matches!(err, DispatchError::LocalDimExceedsLimit { dim: 2, .. }));
    }

    #[test]
    fn dispatch_config_display() {
        let cfg = DispatchConfig::new(&[256], &[32], &a770()).unwrap();
        let s = format!("{cfg}");
        assert!(s.contains("global="));
        assert!(s.contains("local="));
        assert!(s.contains("groups="));
    }

    // -- Integration: dispatch_* output passes validation -------------------

    #[test]
    fn dispatch_1d_output_validates() {
        for n in [1, 7, 16, 255, 1024, 65536, 1_000_000] {
            let (g, l) = dispatch_1d(n, &a770());
            assert!(
                validate_dispatch(&[g], &[l], &a770()).is_ok(),
                "dispatch_1d({n}) produced invalid dispatch g={g} l={l}"
            );
        }
    }

    #[test]
    fn dispatch_2d_output_validates() {
        let shapes =
            [(1, 1), (16, 16), (512, 512), (2048, 2048), (4096, 11008), (1, 128256), (100, 300)];
        for (m, n) in shapes {
            let (g, l) = dispatch_2d(m, n, &a770());
            assert!(
                validate_dispatch(&g, &l, &a770()).is_ok(),
                "dispatch_2d({m},{n}) produced invalid dispatch g={g:?} l={l:?}"
            );
        }
    }
}
