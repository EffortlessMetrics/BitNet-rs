//! OpenCL kernel launcher for Intel Arc A770 GPU compute dispatching.
//!
//! Provides a safe, ergonomic API for dispatching OpenCL kernels with automatic
//! work-size computation, argument binding, and launch validation. All functions
//! are CPU reference implementations that mirror the OpenCL dispatch model.

use std::fmt;

// ---------------------------------------------------------------------------
// Constants — Intel Arc A770 defaults
// ---------------------------------------------------------------------------

/// Default maximum workgroup size for Intel Arc A770.
pub const A770_MAX_WORKGROUP_SIZE: usize = 1024;

/// Default maximum work-item dimensions for OpenCL.
pub const A770_MAX_DIMENSIONS: usize = 3;

/// Preferred workgroup size multiple for Intel Arc A770 (subgroup width).
pub const A770_PREFERRED_WORKGROUP_MULTIPLE: usize = 16;

/// Subgroup sizes supported by Intel Arc A770.
pub const A770_SUBGROUP_SIZES: &[usize] = &[8, 16, 32];

// ---------------------------------------------------------------------------
// KernelArg — kernel argument descriptor
// ---------------------------------------------------------------------------

/// Describes a single argument to an OpenCL kernel.
#[derive(Debug, Clone)]
pub enum KernelArg {
    /// A device buffer argument.
    Buffer {
        /// Raw data to upload.
        data: Vec<u8>,
        /// Size in bytes.
        size: usize,
        /// Whether the buffer is read-only on the device.
        read_only: bool,
    },
    /// A 32-bit signed integer scalar.
    ScalarI32(i32),
    /// A 32-bit unsigned integer scalar.
    ScalarU32(u32),
    /// A 32-bit floating-point scalar.
    ScalarF32(f32),
    /// Local (shared) memory allocation in bytes.
    LocalMem(usize),
}

impl KernelArg {
    /// Size of this argument in bytes for binding purposes.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Buffer { size, .. } => *size,
            Self::ScalarI32(_) => std::mem::size_of::<i32>(),
            Self::ScalarU32(_) => std::mem::size_of::<u32>(),
            Self::ScalarF32(_) => std::mem::size_of::<f32>(),
            Self::LocalMem(sz) => *sz,
        }
    }
}

// ---------------------------------------------------------------------------
// LaunchConfig — work-size configuration
// ---------------------------------------------------------------------------

/// Specifies the global and local work sizes for a kernel dispatch.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Global work size per dimension.
    pub global_work_size: Vec<usize>,
    /// Optional local (workgroup) work size per dimension.
    pub local_work_size: Option<Vec<usize>>,
    /// Number of dispatch dimensions (1, 2, or 3).
    pub dimensions: usize,
}

// ---------------------------------------------------------------------------
// KernelSource — kernel program source
// ---------------------------------------------------------------------------

/// An OpenCL kernel program source and compilation options.
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// Kernel function entry-point name.
    pub name: String,
    /// OpenCL C source code.
    pub source: String,
    /// Compiler build options (e.g., `-cl-fast-relaxed-math`).
    pub build_options: String,
}

// ---------------------------------------------------------------------------
// LaunchRequest — full dispatch descriptor
// ---------------------------------------------------------------------------

/// A complete kernel launch request bundling source, arguments, and config.
#[derive(Debug, Clone)]
pub struct LaunchRequest {
    /// Kernel program to compile and run.
    pub kernel: KernelSource,
    /// Arguments to bind before dispatch.
    pub args: Vec<KernelArg>,
    /// Work-size configuration.
    pub config: LaunchConfig,
    /// Optional profiling event name for timing.
    pub event_name: Option<String>,
}

// ---------------------------------------------------------------------------
// LaunchResult — dispatch outcome
// ---------------------------------------------------------------------------

/// Result of a kernel launch (or simulated launch).
#[derive(Debug, Clone)]
pub struct LaunchResult {
    /// Whether execution completed without error.
    pub success: bool,
    /// Simulated execution time in microseconds.
    pub execution_time_us: u64,
    /// Output buffers read back from device (one per writable buffer arg).
    pub output_data: Vec<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// KernelLauncher — stateful launcher
// ---------------------------------------------------------------------------

/// Manages kernel launch parameters and tracks dispatch statistics.
#[derive(Debug, Clone)]
pub struct KernelLauncher {
    /// Maximum work-items in a single workgroup.
    pub max_workgroup_size: usize,
    /// Maximum number of dispatch dimensions.
    pub max_dimensions: usize,
    /// Hardware preferred workgroup size multiple.
    pub preferred_workgroup_multiple: usize,
    /// Supported subgroup sizes (e.g., 8, 16, 32 for Arc).
    pub subgroup_sizes: Vec<usize>,
    /// Running count of dispatches.
    pub launch_count: u64,
}

// ---------------------------------------------------------------------------
// WorkSizeHint — strategy for local work-size selection
// ---------------------------------------------------------------------------

/// Hint for how to compute the local work size.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkSizeHint {
    /// Let the implementation choose.
    Auto,
    /// Use an explicit local size.
    Explicit(Vec<usize>),
    /// Align the first dimension to the given subgroup width.
    SubgroupAligned(usize),
    /// Maximise occupancy (fill workgroup to hardware limit).
    MaxOccupancy,
}

// ---------------------------------------------------------------------------
// LaunchError — dispatch failures
// ---------------------------------------------------------------------------

/// Errors that can occur during kernel launch validation or dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum LaunchError {
    /// The requested number of dimensions exceeds the device limit.
    InvalidDimensions(usize),
    /// The total workgroup size exceeds the device maximum.
    WorkgroupTooLarge {
        /// Total workgroup size requested.
        requested: usize,
        /// Device maximum.
        max: usize,
    },
    /// Kernel source expected a different number of arguments.
    ArgCountMismatch {
        /// Expected argument count.
        expected: usize,
        /// Actual argument count.
        got: usize,
    },
    /// Kernel source failed to compile.
    KernelCompileFailed(String),
    /// Dispatch failed at runtime.
    DispatchFailed(String),
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(d) => {
                write!(f, "invalid dimensions: {d} (max 3)")
            }
            Self::WorkgroupTooLarge { requested, max } => {
                write!(f, "workgroup too large: {requested} exceeds max {max}")
            }
            Self::ArgCountMismatch { expected, got } => {
                write!(f, "argument count mismatch: expected {expected}, got {got}")
            }
            Self::KernelCompileFailed(msg) => {
                write!(f, "kernel compile failed: {msg}")
            }
            Self::DispatchFailed(msg) => {
                write!(f, "dispatch failed: {msg}")
            }
        }
    }
}

impl std::error::Error for LaunchError {}

// ===========================================================================
// CPU reference implementations
// ===========================================================================

/// Create a [`KernelLauncher`] with the given hardware parameters.
pub fn create_kernel_launcher(max_workgroup: usize, subgroup_sizes: Vec<usize>) -> KernelLauncher {
    let preferred = subgroup_sizes.iter().copied().max().unwrap_or(16);
    KernelLauncher {
        max_workgroup_size: max_workgroup,
        max_dimensions: A770_MAX_DIMENSIONS,
        preferred_workgroup_multiple: preferred,
        subgroup_sizes,
        launch_count: 0,
    }
}

/// Compute an optimal local work size from a [`WorkSizeHint`].
pub fn cpu_compute_local_work_size(
    global: &[usize],
    hint: WorkSizeHint,
    max_wg: usize,
) -> Vec<usize> {
    match hint {
        WorkSizeHint::Explicit(local) => local,
        WorkSizeHint::SubgroupAligned(sg) => {
            let mut local = vec![1; global.len()];
            local[0] = sg.min(max_wg).min(global[0]);
            local
        }
        WorkSizeHint::MaxOccupancy => compute_auto_local(global, max_wg),
        WorkSizeHint::Auto => compute_auto_local(global, max_wg),
    }
}

/// Internal helper: pick local sizes that evenly divide global where possible.
fn compute_auto_local(global: &[usize], max_wg: usize) -> Vec<usize> {
    let dims = global.len();
    let mut local = vec![1; dims];

    if dims == 1 {
        // 1-D: largest divisor of global[0] <= max_wg, preferring
        // multiples of 16 (subgroup width).
        local[0] = best_divisor(global[0], max_wg);
    } else if dims == 2 {
        // 2-D: balance between dimensions.
        let sqrt_max = (max_wg as f64).sqrt() as usize;
        local[0] = best_divisor(global[0], sqrt_max.max(1));
        let remaining = max_wg / local[0].max(1);
        local[1] = best_divisor(global[1], remaining);
    } else if dims >= 3 {
        // 3-D: cube-root split.
        let cbrt_max = (max_wg as f64).cbrt() as usize;
        local[0] = best_divisor(global[0], cbrt_max.max(1));
        let remaining_2d = max_wg / local[0].max(1);
        let sqrt_rem = (remaining_2d as f64).sqrt() as usize;
        local[1] = best_divisor(global[1], sqrt_rem.max(1));
        let remaining_1d = remaining_2d / local[1].max(1);
        local[2] = best_divisor(global[2], remaining_1d);
    }

    local
}

/// Find the largest divisor of `n` that is ≤ `limit`.
fn best_divisor(n: usize, limit: usize) -> usize {
    if n == 0 || limit == 0 {
        return 1;
    }
    let cap = n.min(limit);
    // Try multiples of 16 first (subgroup-friendly).
    let mut best = 1;
    let mut candidate = 16;
    while candidate <= cap {
        if n.is_multiple_of(candidate) {
            best = candidate;
        }
        candidate += 16;
    }
    if best > 1 {
        return best;
    }
    // Fall back to largest divisor <= cap.
    for d in (1..=cap).rev() {
        if n.is_multiple_of(d) {
            return d;
        }
    }
    1
}

/// Round each element of `global` up to the nearest multiple of the
/// corresponding element in `local`.
pub fn cpu_round_up_global(global: &[usize], local: &[usize]) -> Vec<usize> {
    global
        .iter()
        .zip(local.iter())
        .map(|(&g, &l)| {
            if l == 0 {
                return g;
            }
            g.div_ceil(l) * l
        })
        .collect()
}

/// Validate a [`LaunchConfig`] against hardware limits.
pub fn cpu_validate_launch_config(
    config: &LaunchConfig,
    max_wg: usize,
    max_dims: usize,
) -> Result<(), LaunchError> {
    if config.dimensions == 0 || config.dimensions > max_dims {
        return Err(LaunchError::InvalidDimensions(config.dimensions));
    }
    if let Some(ref local) = config.local_work_size {
        let total: usize = local.iter().product();
        if total > max_wg {
            return Err(LaunchError::WorkgroupTooLarge { requested: total, max: max_wg });
        }
    }
    Ok(())
}

/// Compute (offset, size) bindings for each kernel argument.
pub fn cpu_bind_args(args: &[KernelArg]) -> Vec<(usize, usize)> {
    let mut offset = 0usize;
    args.iter()
        .map(|arg| {
            let size = arg.byte_size();
            let binding = (offset, size);
            offset += size;
            binding
        })
        .collect()
}

/// Simulate a kernel launch on the CPU (no actual GPU dispatch).
///
/// Validates the config, binds arguments, increments the launch counter,
/// and returns a synthetic [`LaunchResult`].
pub fn cpu_simulate_launch(
    launcher: &mut KernelLauncher,
    request: &LaunchRequest,
) -> Result<LaunchResult, LaunchError> {
    cpu_validate_launch_config(
        &request.config,
        launcher.max_workgroup_size,
        launcher.max_dimensions,
    )?;

    if request.kernel.source.is_empty() {
        return Err(LaunchError::KernelCompileFailed("empty kernel source".to_string()));
    }

    let _bindings = cpu_bind_args(&request.args);

    // Collect output placeholders for writable buffers.
    let output_data: Vec<Vec<u8>> = request
        .args
        .iter()
        .filter_map(|arg| match arg {
            KernelArg::Buffer { size, read_only: false, .. } => Some(vec![0u8; *size]),
            _ => None,
        })
        .collect();

    let workgroups = cpu_compute_workgroups(
        &request.config.global_work_size,
        request.config.local_work_size.as_deref().unwrap_or(&[1]),
    );

    launcher.launch_count += 1;

    // Synthetic execution time: 1 µs per workgroup.
    let execution_time_us = workgroups as u64;

    Ok(LaunchResult { success: true, execution_time_us, output_data })
}

/// Compute total number of workgroups given global and local sizes.
pub fn cpu_compute_workgroups(global: &[usize], local: &[usize]) -> usize {
    global
        .iter()
        .zip(local.iter())
        .map(|(&g, &l)| {
            if l == 0 {
                return 0;
            }
            g.div_ceil(l)
        })
        .product::<usize>()
        .max(1)
}

/// Estimate occupancy as the fraction of the maximum workgroup size used.
///
/// `shared` is the local memory in bytes consumed by the kernel. The
/// heuristic penalises large register or shared-memory usage.
pub fn cpu_estimate_occupancy(local: &[usize], registers: u32, shared: u32, max_wg: usize) -> f32 {
    let total: usize = local.iter().product();
    if max_wg == 0 {
        return 0.0;
    }
    let base = total as f32 / max_wg as f32;

    // Penalise high register pressure (> 64 regs → linear decay).
    let reg_penalty =
        if registers > 64 { 1.0 - ((registers - 64) as f32 / 256.0).min(0.5) } else { 1.0 };

    // Penalise large shared memory (> 16 KiB → linear decay).
    let shared_penalty =
        if shared > 16_384 { 1.0 - ((shared - 16_384) as f32 / 65_536.0).min(0.5) } else { 1.0 };

    (base * reg_penalty * shared_penalty).clamp(0.0, 1.0)
}

/// Select the best subgroup size for a given problem size.
///
/// Prefers the largest subgroup that evenly divides the problem, falling
/// back to the smallest available subgroup.
pub fn cpu_select_subgroup_size(problem_size: usize, available: &[usize]) -> usize {
    if available.is_empty() {
        return 1;
    }
    // Largest subgroup that divides the problem size.
    available.iter().copied().filter(|&sg| problem_size.is_multiple_of(sg)).max().unwrap_or_else(
        || {
            // Fall back to smallest available.
            available.iter().copied().min().unwrap_or(1)
        },
    )
}

/// Launch multiple kernels sequentially, collecting results.
pub fn cpu_batch_launch(
    launcher: &mut KernelLauncher,
    requests: &[LaunchRequest],
) -> Vec<Result<LaunchResult, LaunchError>> {
    requests.iter().map(|req| cpu_simulate_launch(launcher, req)).collect()
}

/// Return a human-readable summary of launcher statistics.
pub fn cpu_get_launch_stats(launcher: &KernelLauncher) -> String {
    format!(
        "launches={} max_wg={} subgroups={:?}",
        launcher.launch_count, launcher.max_workgroup_size, launcher.subgroup_sizes,
    )
}

/// Format a [`LaunchConfig`] for debug / log output.
pub fn format_launch_config(config: &LaunchConfig) -> String {
    let local_str = match &config.local_work_size {
        Some(l) => format!("{l:?}"),
        None => "auto".to_string(),
    };
    format!("{}D global={:?} local={}", config.dimensions, config.global_work_size, local_str,)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn a770_launcher() -> KernelLauncher {
        create_kernel_launcher(A770_MAX_WORKGROUP_SIZE, A770_SUBGROUP_SIZES.to_vec())
    }

    fn simple_request(global: Vec<usize>) -> LaunchRequest {
        LaunchRequest {
            kernel: KernelSource {
                name: "test_kernel".into(),
                source: "__kernel void test_kernel() {}".into(),
                build_options: String::new(),
            },
            args: vec![
                KernelArg::Buffer { data: vec![0u8; 64], size: 64, read_only: true },
                KernelArg::ScalarI32(42),
            ],
            config: LaunchConfig {
                global_work_size: global.clone(),
                local_work_size: None,
                dimensions: global.len(),
            },
            event_name: None,
        }
    }

    // -----------------------------------------------------------------------
    // 1. Create launcher with A770 defaults
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_launcher_a770_defaults() {
        let l = a770_launcher();
        assert_eq!(l.max_workgroup_size, 1024);
        assert_eq!(l.max_dimensions, 3);
        assert_eq!(l.preferred_workgroup_multiple, 32); // max of [8,16,32]
        assert_eq!(l.subgroup_sizes, vec![8, 16, 32]);
        assert_eq!(l.launch_count, 0);
    }

    // -----------------------------------------------------------------------
    // 2-4. Compute local size: auto for 1D, 2D, 3D
    // -----------------------------------------------------------------------
    #[test]
    fn test_auto_local_1d() {
        let local = cpu_compute_local_work_size(&[1024], WorkSizeHint::Auto, 256);
        assert_eq!(local.len(), 1);
        assert!(local[0] > 0);
        assert!(local[0] <= 256);
        assert_eq!(1024 % local[0], 0);
    }

    #[test]
    fn test_auto_local_2d() {
        let local = cpu_compute_local_work_size(&[256, 256], WorkSizeHint::Auto, 256);
        assert_eq!(local.len(), 2);
        let total: usize = local.iter().product();
        assert!(total <= 256);
    }

    #[test]
    fn test_auto_local_3d() {
        let local = cpu_compute_local_work_size(&[64, 64, 64], WorkSizeHint::Auto, 256);
        assert_eq!(local.len(), 3);
        let total: usize = local.iter().product();
        assert!(total <= 256);
    }

    // -----------------------------------------------------------------------
    // 5. Explicit local work size
    // -----------------------------------------------------------------------
    #[test]
    fn test_explicit_local() {
        let local = cpu_compute_local_work_size(&[1024], WorkSizeHint::Explicit(vec![64]), 1024);
        assert_eq!(local, vec![64]);
    }

    // -----------------------------------------------------------------------
    // 6. Subgroup-aligned hint
    // -----------------------------------------------------------------------
    #[test]
    fn test_subgroup_aligned_hint() {
        let local =
            cpu_compute_local_work_size(&[1024, 512], WorkSizeHint::SubgroupAligned(16), 1024);
        assert_eq!(local[0], 16);
        assert_eq!(local[1], 1);
    }

    // -----------------------------------------------------------------------
    // 7. MaxOccupancy hint
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_occupancy_hint() {
        let local = cpu_compute_local_work_size(&[1024], WorkSizeHint::MaxOccupancy, 1024);
        assert!(local[0] > 0);
        assert!(local[0] <= 1024);
    }

    // -----------------------------------------------------------------------
    // 8-9. Round up global: correct multiples
    // -----------------------------------------------------------------------
    #[test]
    fn test_round_up_global_exact() {
        let rounded = cpu_round_up_global(&[1024], &[256]);
        assert_eq!(rounded, vec![1024]);
    }

    #[test]
    fn test_round_up_global_inexact() {
        let rounded = cpu_round_up_global(&[1000], &[256]);
        assert_eq!(rounded, vec![1024]);
    }

    // -----------------------------------------------------------------------
    // 10-11. Round up global: multi-dimensional
    // -----------------------------------------------------------------------
    #[test]
    fn test_round_up_global_2d() {
        let rounded = cpu_round_up_global(&[100, 200], &[16, 16]);
        assert_eq!(rounded, vec![112, 208]);
    }

    #[test]
    fn test_round_up_global_3d() {
        let rounded = cpu_round_up_global(&[30, 30, 30], &[8, 8, 8]);
        assert_eq!(rounded, vec![32, 32, 32]);
    }

    // -----------------------------------------------------------------------
    // 12. Validate config: valid passes
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_config_valid() {
        let config = LaunchConfig {
            global_work_size: vec![1024],
            local_work_size: Some(vec![256]),
            dimensions: 1,
        };
        assert!(cpu_validate_launch_config(&config, 1024, 3).is_ok());
    }

    // -----------------------------------------------------------------------
    // 13. Validate config: too-large workgroup fails
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_workgroup_too_large() {
        let config = LaunchConfig {
            global_work_size: vec![4096],
            local_work_size: Some(vec![2048]),
            dimensions: 1,
        };
        let err = cpu_validate_launch_config(&config, 1024, 3).unwrap_err();
        assert_eq!(err, LaunchError::WorkgroupTooLarge { requested: 2048, max: 1024 });
    }

    // -----------------------------------------------------------------------
    // 14. Validate config: dimension limit exceeded
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_invalid_dimensions() {
        let config =
            LaunchConfig { global_work_size: vec![64; 4], local_work_size: None, dimensions: 4 };
        let err = cpu_validate_launch_config(&config, 1024, 3).unwrap_err();
        assert_eq!(err, LaunchError::InvalidDimensions(4));
    }

    // -----------------------------------------------------------------------
    // 15. Validate config: zero dimensions
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_zero_dimensions() {
        let config =
            LaunchConfig { global_work_size: vec![], local_work_size: None, dimensions: 0 };
        assert_eq!(
            cpu_validate_launch_config(&config, 1024, 3).unwrap_err(),
            LaunchError::InvalidDimensions(0),
        );
    }

    // -----------------------------------------------------------------------
    // 16-17. Bind args: correct offsets
    // -----------------------------------------------------------------------
    #[test]
    fn test_bind_args_offsets() {
        let args = vec![
            KernelArg::Buffer { data: vec![0; 128], size: 128, read_only: true },
            KernelArg::ScalarI32(1),
            KernelArg::ScalarF32(1.0),
        ];
        let bindings = cpu_bind_args(&args);
        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0], (0, 128));
        assert_eq!(bindings[1], (128, 4));
        assert_eq!(bindings[2], (132, 4));
    }

    #[test]
    fn test_bind_args_empty() {
        let bindings = cpu_bind_args(&[]);
        assert!(bindings.is_empty());
    }

    // -----------------------------------------------------------------------
    // 18. Bind args: local memory argument
    // -----------------------------------------------------------------------
    #[test]
    fn test_bind_args_local_mem() {
        let args = vec![KernelArg::ScalarU32(10), KernelArg::LocalMem(4096)];
        let bindings = cpu_bind_args(&args);
        assert_eq!(bindings[0], (0, 4));
        assert_eq!(bindings[1], (4, 4096));
    }

    // -----------------------------------------------------------------------
    // 19. Simulate launch: succeeds
    // -----------------------------------------------------------------------
    #[test]
    fn test_simulate_launch_success() {
        let mut l = a770_launcher();
        let req = simple_request(vec![1024]);
        let result = cpu_simulate_launch(&mut l, &req).unwrap();
        assert!(result.success);
        assert!(result.execution_time_us > 0);
        assert_eq!(l.launch_count, 1);
    }

    // -----------------------------------------------------------------------
    // 20. Simulate launch: empty source fails
    // -----------------------------------------------------------------------
    #[test]
    fn test_simulate_launch_empty_source() {
        let mut l = a770_launcher();
        let req = LaunchRequest {
            kernel: KernelSource {
                name: "bad".into(),
                source: String::new(),
                build_options: String::new(),
            },
            args: vec![],
            config: LaunchConfig {
                global_work_size: vec![64],
                local_work_size: None,
                dimensions: 1,
            },
            event_name: None,
        };
        let err = cpu_simulate_launch(&mut l, &req).unwrap_err();
        assert!(matches!(err, LaunchError::KernelCompileFailed(_)));
    }

    // -----------------------------------------------------------------------
    // 21. Simulate launch: output buffers collected
    // -----------------------------------------------------------------------
    #[test]
    fn test_simulate_launch_output_buffers() {
        let mut l = a770_launcher();
        let req = LaunchRequest {
            kernel: KernelSource {
                name: "k".into(),
                source: "__kernel void k() {}".into(),
                build_options: String::new(),
            },
            args: vec![
                KernelArg::Buffer { data: vec![0; 32], size: 32, read_only: true },
                KernelArg::Buffer { data: vec![0; 64], size: 64, read_only: false },
                KernelArg::Buffer { data: vec![0; 16], size: 16, read_only: false },
            ],
            config: LaunchConfig {
                global_work_size: vec![256],
                local_work_size: None,
                dimensions: 1,
            },
            event_name: None,
        };
        let result = cpu_simulate_launch(&mut l, &req).unwrap();
        // Two writable buffers → two output entries.
        assert_eq!(result.output_data.len(), 2);
        assert_eq!(result.output_data[0].len(), 64);
        assert_eq!(result.output_data[1].len(), 16);
    }

    // -----------------------------------------------------------------------
    // 22. Workgroup count: correct math
    // -----------------------------------------------------------------------
    #[test]
    fn test_workgroup_count_1d() {
        assert_eq!(cpu_compute_workgroups(&[1024], &[256]), 4);
    }

    // -----------------------------------------------------------------------
    // 23. Workgroup count: 2D
    // -----------------------------------------------------------------------
    #[test]
    fn test_workgroup_count_2d() {
        assert_eq!(cpu_compute_workgroups(&[1024, 512], &[256, 128]), 4 * 4);
    }

    // -----------------------------------------------------------------------
    // 24. Workgroup count: inexact division rounds up
    // -----------------------------------------------------------------------
    #[test]
    fn test_workgroup_count_round_up() {
        // 1000 / 256 → 4 workgroups (rounded up)
        assert_eq!(cpu_compute_workgroups(&[1000], &[256]), 4);
    }

    // -----------------------------------------------------------------------
    // 25. Occupancy estimate: reasonable range
    // -----------------------------------------------------------------------
    #[test]
    fn test_occupancy_basic() {
        let occ = cpu_estimate_occupancy(&[256], 32, 0, 1024);
        assert!((0.0..=1.0).contains(&occ));
        assert!((occ - 0.25).abs() < 0.01); // 256/1024
    }

    // -----------------------------------------------------------------------
    // 26. Occupancy: register pressure penalty
    // -----------------------------------------------------------------------
    #[test]
    fn test_occupancy_register_penalty() {
        let baseline = cpu_estimate_occupancy(&[512], 32, 0, 1024);
        let penalised = cpu_estimate_occupancy(&[512], 128, 0, 1024);
        assert!(penalised < baseline);
    }

    // -----------------------------------------------------------------------
    // 27. Occupancy: shared memory penalty
    // -----------------------------------------------------------------------
    #[test]
    fn test_occupancy_shared_penalty() {
        let baseline = cpu_estimate_occupancy(&[512], 32, 0, 1024);
        let penalised = cpu_estimate_occupancy(&[512], 32, 32_768, 1024);
        assert!(penalised < baseline);
    }

    // -----------------------------------------------------------------------
    // 28. Subgroup size selection: picks optimal
    // -----------------------------------------------------------------------
    #[test]
    fn test_subgroup_selection_divisible() {
        let sg = cpu_select_subgroup_size(256, &[8, 16, 32]);
        assert_eq!(sg, 32); // 256 divisible by all; pick largest
    }

    // -----------------------------------------------------------------------
    // 29. Subgroup size selection: non-divisible fallback
    // -----------------------------------------------------------------------
    #[test]
    fn test_subgroup_selection_non_divisible() {
        // 17 not divisible by 8, 16, or 32 → fallback to smallest.
        let sg = cpu_select_subgroup_size(17, &[8, 16, 32]);
        assert_eq!(sg, 8);
    }

    // -----------------------------------------------------------------------
    // 30. Subgroup selection: partial divisibility
    // -----------------------------------------------------------------------
    #[test]
    fn test_subgroup_selection_partial() {
        // 48 is divisible by 8 and 16 but not 32.
        let sg = cpu_select_subgroup_size(48, &[8, 16, 32]);
        assert_eq!(sg, 16);
    }

    // -----------------------------------------------------------------------
    // 31. Batch launch: multiple kernels
    // -----------------------------------------------------------------------
    #[test]
    fn test_batch_launch() {
        let mut l = a770_launcher();
        let requests =
            vec![simple_request(vec![1024]), simple_request(vec![512]), simple_request(vec![256])];
        let results = cpu_batch_launch(&mut l, &requests);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok());
        }
        assert_eq!(l.launch_count, 3);
    }

    // -----------------------------------------------------------------------
    // 32. Batch launch: partial failure
    // -----------------------------------------------------------------------
    #[test]
    fn test_batch_launch_partial_failure() {
        let mut l = a770_launcher();
        let mut bad = simple_request(vec![64; 4]); // 4D → invalid
        bad.config.dimensions = 4;
        let requests = vec![simple_request(vec![256]), bad];
        let results = cpu_batch_launch(&mut l, &requests);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }

    // -----------------------------------------------------------------------
    // 33. Edge: global size 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_edge_global_size_one() {
        let mut l = a770_launcher();
        let req = simple_request(vec![1]);
        let result = cpu_simulate_launch(&mut l, &req).unwrap();
        assert!(result.success);
    }

    // -----------------------------------------------------------------------
    // 34. Edge: 3D dispatch
    // -----------------------------------------------------------------------
    #[test]
    fn test_edge_3d_dispatch() {
        let mut l = a770_launcher();
        let req = simple_request(vec![64, 64, 64]);
        let result = cpu_simulate_launch(&mut l, &req).unwrap();
        assert!(result.success);
    }

    // -----------------------------------------------------------------------
    // 35. Edge: local == global
    // -----------------------------------------------------------------------
    #[test]
    fn test_edge_local_equals_global() {
        let rounded = cpu_round_up_global(&[256], &[256]);
        assert_eq!(rounded, vec![256]);
    }

    // -----------------------------------------------------------------------
    // 36. Property: rounded global >= original
    // -----------------------------------------------------------------------
    #[test]
    fn test_property_rounded_geq_original() {
        for g in [1, 7, 15, 33, 100, 1023, 4096] {
            for l in [1, 8, 16, 32, 64, 256] {
                let rounded = cpu_round_up_global(&[g], &[l]);
                assert!(rounded[0] >= g, "rounded {rounded:?} < original {g} with local {l}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // 37. Property: workgroup count >= 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_property_workgroup_count_positive() {
        for g in [1, 7, 64, 1024] {
            for l in [1, 8, 32, 64] {
                assert!(cpu_compute_workgroups(&[g], &[l]) >= 1, "wg count < 1 for g={g} l={l}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // 38. A770: workgroup 1024 limit
    // -----------------------------------------------------------------------
    #[test]
    fn test_a770_workgroup_1024_limit() {
        let config = LaunchConfig {
            global_work_size: vec![4096],
            local_work_size: Some(vec![1024]),
            dimensions: 1,
        };
        assert!(
            cpu_validate_launch_config(&config, A770_MAX_WORKGROUP_SIZE, A770_MAX_DIMENSIONS,)
                .is_ok()
        );
    }

    // -----------------------------------------------------------------------
    // 39. A770: workgroup 1025 exceeds limit
    // -----------------------------------------------------------------------
    #[test]
    fn test_a770_workgroup_exceeds_limit() {
        let config = LaunchConfig {
            global_work_size: vec![4096],
            local_work_size: Some(vec![1025]),
            dimensions: 1,
        };
        assert!(
            cpu_validate_launch_config(&config, A770_MAX_WORKGROUP_SIZE, A770_MAX_DIMENSIONS,)
                .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // 40. A770: subgroup sizes 8/16/32
    // -----------------------------------------------------------------------
    #[test]
    fn test_a770_subgroup_sizes() {
        let l = a770_launcher();
        assert_eq!(l.subgroup_sizes, vec![8, 16, 32]);
    }

    // -----------------------------------------------------------------------
    // 41. Launch stats formatting
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_launch_stats() {
        let mut l = a770_launcher();
        let _ = cpu_simulate_launch(&mut l, &simple_request(vec![128]));
        let stats = cpu_get_launch_stats(&l);
        assert!(stats.contains("launches=1"));
        assert!(stats.contains("max_wg=1024"));
    }

    // -----------------------------------------------------------------------
    // 42. Format launch config
    // -----------------------------------------------------------------------
    #[test]
    fn test_format_launch_config() {
        let config = LaunchConfig {
            global_work_size: vec![1024, 512],
            local_work_size: Some(vec![16, 16]),
            dimensions: 2,
        };
        let s = format_launch_config(&config);
        assert!(s.contains("2D"));
        assert!(s.contains("[1024, 512]"));
        assert!(s.contains("[16, 16]"));
    }

    // -----------------------------------------------------------------------
    // 43. Format launch config with auto local
    // -----------------------------------------------------------------------
    #[test]
    fn test_format_launch_config_auto() {
        let config =
            LaunchConfig { global_work_size: vec![256], local_work_size: None, dimensions: 1 };
        let s = format_launch_config(&config);
        assert!(s.contains("auto"));
    }

    // -----------------------------------------------------------------------
    // 44. Subgroup selection: empty list
    // -----------------------------------------------------------------------
    #[test]
    fn test_subgroup_selection_empty() {
        assert_eq!(cpu_select_subgroup_size(128, &[]), 1);
    }

    // -----------------------------------------------------------------------
    // 45. KernelArg byte sizes
    // -----------------------------------------------------------------------
    #[test]
    fn test_kernel_arg_byte_sizes() {
        assert_eq!(KernelArg::ScalarI32(0).byte_size(), 4);
        assert_eq!(KernelArg::ScalarU32(0).byte_size(), 4);
        assert_eq!(KernelArg::ScalarF32(0.0).byte_size(), 4);
        assert_eq!(KernelArg::LocalMem(2048).byte_size(), 2048);
        assert_eq!(KernelArg::Buffer { data: vec![], size: 512, read_only: true }.byte_size(), 512,);
    }

    // -----------------------------------------------------------------------
    // 46. LaunchError display
    // -----------------------------------------------------------------------
    #[test]
    fn test_launch_error_display() {
        let e = LaunchError::InvalidDimensions(5);
        assert!(e.to_string().contains("5"));
        let e = LaunchError::WorkgroupTooLarge { requested: 2048, max: 1024 };
        assert!(e.to_string().contains("2048"));
    }

    // -----------------------------------------------------------------------
    // 47. Round up with local size zero (no panic)
    // -----------------------------------------------------------------------
    #[test]
    fn test_round_up_local_zero() {
        let rounded = cpu_round_up_global(&[100], &[0]);
        assert_eq!(rounded, vec![100]);
    }

    // -----------------------------------------------------------------------
    // 48. Occupancy: max_wg zero returns 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_occupancy_max_wg_zero() {
        assert_eq!(cpu_estimate_occupancy(&[64], 32, 0, 0), 0.0);
    }

    // -----------------------------------------------------------------------
    // 49. Validate config: 2D workgroup product
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_2d_workgroup_product() {
        let config = LaunchConfig {
            global_work_size: vec![512, 512],
            local_work_size: Some(vec![32, 32]), // product = 1024 (ok)
            dimensions: 2,
        };
        assert!(cpu_validate_launch_config(&config, 1024, 3).is_ok());

        let config2 = LaunchConfig {
            global_work_size: vec![512, 512],
            local_work_size: Some(vec![32, 64]), // product = 2048 (too big)
            dimensions: 2,
        };
        assert!(cpu_validate_launch_config(&config2, 1024, 3).is_err());
    }

    // -----------------------------------------------------------------------
    // 50. Multiple launches increment counter
    // -----------------------------------------------------------------------
    #[test]
    fn test_launch_count_increments() {
        let mut l = a770_launcher();
        for i in 0..5 {
            let _ = cpu_simulate_launch(&mut l, &simple_request(vec![64]));
            assert_eq!(l.launch_count, i + 1);
        }
    }
}
