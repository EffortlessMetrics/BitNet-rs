//! Scatter/gather operations for indexed tensor access.
//!
//! # Operations
//!
//! - **Gather**: Select elements from `src` at `indices` along a given axis,
//!   producing a tensor whose shape matches `indices` on that axis.
//! - **Scatter**: Place elements from `src` into `dst` at `indices` along a
//!   given axis, optionally applying a reduction (assign, add, max, min).
//! - **Index select**: Simplified 1-D gather along axis 0 — selects full rows
//!   from a 2-D matrix.
//!
//! # CPU fallback
//!
//! All three operations have pure-Rust CPU implementations for correctness
//! testing and non-GPU environments.  The unified dispatch functions
//! ([`gather_forward`], [`scatter_forward`]) try the GPU path first when
//! compiled with `gpu`/`cuda` features and fall back to CPU otherwise.
//!
//! # CUDA kernel stubs
//!
//! GPU launch stubs are gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`
//! and return `KernelError::GpuError` until real PTX kernels are compiled.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Enums and configuration
// ---------------------------------------------------------------------------

/// Reduction mode applied when scattering overlapping indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterMode {
    /// Overwrite the destination element (last write wins).
    Assign,
    /// Add the source value to the destination element.
    Add,
    /// Keep the maximum of source and destination.
    Max,
    /// Keep the minimum of source and destination.
    Min,
}

impl ScatterMode {
    /// Return the identity element for the reduction so that
    /// `combine(identity, x) == x` for every `x`.
    ///
    /// Useful for pre-filling destination buffers before scatter.
    pub fn identity(self) -> f32 {
        match self {
            Self::Assign | Self::Add => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
        }
    }

    /// Combine two values according to the reduction mode.
    fn combine(self, dst: f32, src: f32) -> f32 {
        match self {
            Self::Assign => src,
            Self::Add => dst + src,
            Self::Max => dst.max(src),
            Self::Min => dst.min(src),
        }
    }
}

/// Configuration for gather/scatter operations.
#[derive(Debug, Clone)]
pub struct GatherConfig {
    /// Axis along which to index (0 = rows, 1 = cols for 2-D tensors).
    pub axis: usize,
    /// Shape of the index tensor `[index_rows, index_cols]`.
    pub indices_shape: (usize, usize),
    /// When `true`, out-of-bounds indices return an error instead of
    /// silently clamping.
    pub bounds_check: bool,
}

impl GatherConfig {
    /// Create a new configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `axis > 1` (only 2-D
    /// tensors are currently supported).
    pub fn new(axis: usize, indices_shape: (usize, usize), bounds_check: bool) -> Result<Self> {
        if axis > 1 {
            return Err(KernelError::InvalidArguments {
                reason: format!("scatter/gather axis must be 0 or 1 for 2-D tensors, got {axis}"),
            }
            .into());
        }
        Ok(Self { axis, indices_shape, bounds_check })
    }
}

/// Kernel handle for scatter/gather operations.
#[derive(Debug, Clone)]
pub struct ScatterGatherKernel {
    /// Source tensor shape `[rows, cols]`.
    pub src_shape: (usize, usize),
    /// Threads per block for GPU launch.
    pub threads_per_block: u32,
}

impl ScatterGatherKernel {
    /// Create a kernel handle for a source tensor of the given shape.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if either dimension is zero.
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "scatter/gather source dimensions must be non-zero: \
                     rows={rows}, cols={cols}"
                ),
            }
            .into());
        }
        let threads_per_block = (cols as u32).min(1024);
        Ok(Self { src_shape: (rows, cols), threads_per_block })
    }

    /// Compute the CUDA grid dimensions for a given number of output elements.
    pub fn grid_dim(&self, n_elements: usize) -> (u32, u32, u32) {
        let blocks = (n_elements as u32).div_ceil(self.threads_per_block);
        (blocks.max(1), 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// CPU fallback — gather
// ---------------------------------------------------------------------------

/// Gather elements from `src` at positions given by `indices` along `axis`.
///
/// For a 2-D source `[S_rows, S_cols]` and index matrix `[I_rows, I_cols]`:
///
/// - **axis 0**: `output[i][j] = src[indices[i][j]][j]`
///   — each index selects a *row* to read from.
///   Requires `I_cols == S_cols`.
/// - **axis 1**: `output[i][j] = src[i][indices[i][j]]`
///   — each index selects a *column* to read from.
///   Requires `I_rows == S_rows`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on shape mismatch or
/// out-of-bounds indices (when `config.bounds_check` is true).
pub fn gather_cpu(
    src: &[f32],
    indices: &[usize],
    output: &mut [f32],
    kernel: &ScatterGatherKernel,
    config: &GatherConfig,
) -> Result<()> {
    let (s_rows, s_cols) = kernel.src_shape;
    let (i_rows, i_cols) = config.indices_shape;
    let out_len = i_rows * i_cols;

    validate_gather_shapes(s_rows, s_cols, i_rows, i_cols, config.axis)?;

    if indices.len() < out_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("gather indices length {} < expected {}", indices.len(), out_len,),
        }
        .into());
    }
    if output.len() < out_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("gather output length {} < expected {}", output.len(), out_len,),
        }
        .into());
    }

    let bound = if config.axis == 0 { s_rows } else { s_cols };

    for i in 0..i_rows {
        for j in 0..i_cols {
            let idx = indices[i * i_cols + j];
            if config.bounds_check && idx >= bound {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "gather index {idx} out of bounds for axis {} \
                         with size {bound}",
                        config.axis,
                    ),
                }
                .into());
            }
            let clamped = idx.min(bound.saturating_sub(1));
            let src_offset =
                if config.axis == 0 { clamped * s_cols + j } else { i * s_cols + clamped };
            output[i * i_cols + j] = src[src_offset];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — scatter
// ---------------------------------------------------------------------------

/// Scatter elements from `src` into `dst` at positions given by `indices`
/// along `axis`, applying the specified `mode` reduction.
///
/// For a 2-D destination `[D_rows, D_cols]` and source `[I_rows, I_cols]`:
///
/// - **axis 0**: `dst[indices[i][j]][j] = mode(dst[…], src[i][j])`
/// - **axis 1**: `dst[i][indices[i][j]] = mode(dst[…], src[i][j])`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on shape mismatch or
/// out-of-bounds indices (when `config.bounds_check` is true).
pub fn scatter_cpu(
    src: &[f32],
    indices: &[usize],
    dst: &mut [f32],
    dst_shape: (usize, usize),
    config: &GatherConfig,
    mode: ScatterMode,
) -> Result<()> {
    let (d_rows, d_cols) = dst_shape;
    let (i_rows, i_cols) = config.indices_shape;
    let elem_count = i_rows * i_cols;

    validate_scatter_shapes(d_rows, d_cols, i_rows, i_cols, config.axis)?;

    if src.len() < elem_count {
        return Err(KernelError::InvalidArguments {
            reason: format!("scatter src length {} < expected {}", src.len(), elem_count,),
        }
        .into());
    }
    if indices.len() < elem_count {
        return Err(KernelError::InvalidArguments {
            reason: format!("scatter indices length {} < expected {}", indices.len(), elem_count,),
        }
        .into());
    }
    if dst.len() < d_rows * d_cols {
        return Err(KernelError::InvalidArguments {
            reason: format!("scatter dst length {} < expected {}", dst.len(), d_rows * d_cols,),
        }
        .into());
    }

    let bound = if config.axis == 0 { d_rows } else { d_cols };

    for i in 0..i_rows {
        for j in 0..i_cols {
            let idx = indices[i * i_cols + j];
            if config.bounds_check && idx >= bound {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "scatter index {idx} out of bounds for axis {} \
                         with size {bound}",
                        config.axis,
                    ),
                }
                .into());
            }
            let clamped = idx.min(bound.saturating_sub(1));
            let dst_offset =
                if config.axis == 0 { clamped * d_cols + j } else { i * d_cols + clamped };
            let src_val = src[i * i_cols + j];
            dst[dst_offset] = mode.combine(dst[dst_offset], src_val);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — index_select
// ---------------------------------------------------------------------------

/// Simplified gather: select full rows from a 2-D matrix by a 1-D index
/// vector.
///
/// Given `src [S_rows, S_cols]` and `indices [N]`, produces
/// `output [N, S_cols]` where `output[i] = src[indices[i]]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch or
/// out-of-bounds indices (when `bounds_check` is true).
pub fn index_select_cpu(
    src: &[f32],
    indices: &[usize],
    output: &mut [f32],
    kernel: &ScatterGatherKernel,
    bounds_check: bool,
) -> Result<()> {
    let (s_rows, s_cols) = kernel.src_shape;
    let n = indices.len();
    let out_len = n * s_cols;

    if src.len() < s_rows * s_cols {
        return Err(
            KernelError::InvalidArguments {
                reason: format!(
                    "index_select src length {} < expected {}",
                    src.len(),
                    s_rows * s_cols,
                ),
            }
            .into(),
        );
    }
    if output.len() < out_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("index_select output length {} < expected {}", output.len(), out_len,),
        }
        .into());
    }

    for (out_row, &idx) in indices.iter().enumerate() {
        if bounds_check && idx >= s_rows {
            return Err(KernelError::InvalidArguments {
                reason: format!("index_select index {idx} out of bounds for {} rows", s_rows,),
            }
            .into());
        }
        let clamped = idx.min(s_rows.saturating_sub(1));
        let src_start = clamped * s_cols;
        let dst_start = out_row * s_cols;
        output[dst_start..dst_start + s_cols].copy_from_slice(&src[src_start..src_start + s_cols]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA launch stubs
// ---------------------------------------------------------------------------

/// GPU launch stub for gather.
///
/// Returns `KernelError::GpuError` until real PTX is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_gather(
    _src: &[f32],
    _indices: &[usize],
    _output: &mut [f32],
    kernel: &ScatterGatherKernel,
    config: &GatherConfig,
) -> Result<()> {
    log::debug!(
        "gather stub: src_shape={:?}, indices_shape={:?}, axis={}, grid={:?}",
        kernel.src_shape,
        config.indices_shape,
        config.axis,
        kernel.grid_dim(config.indices_shape.0 * config.indices_shape.1),
    );
    Err(KernelError::GpuError {
        reason: "gather CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// GPU launch stub for scatter.
///
/// Returns `KernelError::GpuError` until real PTX is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_scatter(
    _src: &[f32],
    _indices: &[usize],
    _dst: &mut [f32],
    _dst_shape: (usize, usize),
    config: &GatherConfig,
    mode: ScatterMode,
) -> Result<()> {
    log::debug!(
        "scatter stub: indices_shape={:?}, axis={}, mode={mode:?}",
        config.indices_shape,
        config.axis,
    );
    Err(KernelError::GpuError {
        reason: "scatter CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Gather with automatic dispatch: GPU if available, else CPU fallback.
pub fn gather_forward(
    src: &[f32],
    indices: &[usize],
    output: &mut [f32],
    kernel: &ScatterGatherKernel,
    config: &GatherConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            if let Ok(()) = launch_gather(src, indices, output, kernel, config) {
                return Ok(());
            }
        }
    }
    gather_cpu(src, indices, output, kernel, config)
}

/// Scatter with automatic dispatch: GPU if available, else CPU fallback.
pub fn scatter_forward(
    src: &[f32],
    indices: &[usize],
    dst: &mut [f32],
    dst_shape: (usize, usize),
    config: &GatherConfig,
    mode: ScatterMode,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            if let Ok(()) = launch_scatter(src, indices, dst, dst_shape, config, mode) {
                return Ok(());
            }
        }
    }
    scatter_cpu(src, indices, dst, dst_shape, config, mode)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_gather_shapes(
    s_rows: usize,
    s_cols: usize,
    i_rows: usize,
    i_cols: usize,
    axis: usize,
) -> Result<()> {
    if axis == 0 && i_cols != s_cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gather axis 0 requires indices cols ({i_cols}) \
                 == src cols ({s_cols})"
            ),
        }
        .into());
    }
    if axis == 1 && i_rows != s_rows {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gather axis 1 requires indices rows ({i_rows}) \
                 == src rows ({s_rows})"
            ),
        }
        .into());
    }
    Ok(())
}

fn validate_scatter_shapes(
    d_rows: usize,
    d_cols: usize,
    i_rows: usize,
    i_cols: usize,
    axis: usize,
) -> Result<()> {
    if axis == 0 && i_cols != d_cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "scatter axis 0 requires indices cols ({i_cols}) \
                 == dst cols ({d_cols})"
            ),
        }
        .into());
    }
    if axis == 1 && i_rows != d_rows {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "scatter axis 1 requires indices rows ({i_rows}) \
                 == dst rows ({d_rows})"
            ),
        }
        .into());
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_gather_config_new() {
        let cfg = GatherConfig::new(0, (3, 4), true).unwrap();
        assert_eq!(cfg.axis, 0);
        assert_eq!(cfg.indices_shape, (3, 4));
        assert!(cfg.bounds_check);
    }

    #[test]
    fn test_gather_config_rejects_axis_gt_1() {
        assert!(GatherConfig::new(2, (3, 4), true).is_err());
        assert!(GatherConfig::new(99, (1, 1), false).is_err());
    }

    #[test]
    fn test_scatter_gather_kernel_new() {
        let k = ScatterGatherKernel::new(8, 64).unwrap();
        assert_eq!(k.src_shape, (8, 64));
        assert_eq!(k.threads_per_block, 64);
    }

    #[test]
    fn test_kernel_rejects_zero_dims() {
        assert!(ScatterGatherKernel::new(0, 4).is_err());
        assert!(ScatterGatherKernel::new(4, 0).is_err());
    }

    #[test]
    fn test_kernel_threads_capped() {
        let k = ScatterGatherKernel::new(1, 4096).unwrap();
        assert_eq!(k.threads_per_block, 1024);
    }

    #[test]
    fn test_kernel_grid_dim() {
        let k = ScatterGatherKernel::new(4, 256).unwrap();
        // 1024 elements, 256 threads/block → 4 blocks
        assert_eq!(k.grid_dim(1024), (4, 1, 1));
        // 1 element → at least 1 block
        assert_eq!(k.grid_dim(1), (1, 1, 1));
    }

    // -- ScatterMode tests --------------------------------------------------

    #[test]
    fn test_scatter_mode_identity() {
        assert_eq!(ScatterMode::Assign.identity(), 0.0);
        assert_eq!(ScatterMode::Add.identity(), 0.0);
        assert_eq!(ScatterMode::Max.identity(), f32::NEG_INFINITY);
        assert_eq!(ScatterMode::Min.identity(), f32::INFINITY);
    }

    #[test]
    fn test_scatter_mode_combine() {
        assert_eq!(ScatterMode::Assign.combine(10.0, 5.0), 5.0);
        assert_eq!(ScatterMode::Add.combine(10.0, 5.0), 15.0);
        assert_eq!(ScatterMode::Max.combine(10.0, 5.0), 10.0);
        assert_eq!(ScatterMode::Min.combine(10.0, 5.0), 5.0);
    }

    // -- Gather CPU tests ---------------------------------------------------

    #[test]
    fn test_gather_axis0_basic() {
        // src: 3×2 matrix, select rows by index
        //   [[10, 11], [20, 21], [30, 31]]
        let src = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let indices = [2, 0]; // select row 2 col0, row 0 col1
        let kernel = ScatterGatherKernel::new(3, 2).unwrap();
        let config = GatherConfig::new(0, (1, 2), true).unwrap();
        let mut output = [0.0_f32; 2];
        gather_cpu(&src, &indices, &mut output, &kernel, &config).unwrap();
        assert_eq!(output, [30.0, 11.0]);
    }

    #[test]
    fn test_gather_axis1_basic() {
        // src: 2×4, select columns by index
        let src: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let indices = [3, 1, 0, 2]; // 2×2
        let kernel = ScatterGatherKernel::new(2, 4).unwrap();
        let config = GatherConfig::new(1, (2, 2), true).unwrap();
        let mut output = [0.0_f32; 4];
        gather_cpu(&src, &indices, &mut output, &kernel, &config).unwrap();
        assert_eq!(output, [3.0, 1.0, 4.0, 6.0]);
    }

    #[test]
    fn test_gather_out_of_bounds_error() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [5, 0]; // 5 is OOB for 2 rows
        let kernel = ScatterGatherKernel::new(2, 2).unwrap();
        let config = GatherConfig::new(0, (1, 2), true).unwrap();
        let mut output = [0.0_f32; 2];
        assert!(gather_cpu(&src, &indices, &mut output, &kernel, &config).is_err());
    }

    #[test]
    fn test_gather_out_of_bounds_clamp() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [5, 0]; // OOB but bounds_check=false → clamp
        let kernel = ScatterGatherKernel::new(2, 2).unwrap();
        let config = GatherConfig::new(0, (1, 2), false).unwrap();
        let mut output = [0.0_f32; 2];
        gather_cpu(&src, &indices, &mut output, &kernel, &config).unwrap();
        // index 5 clamped to row 1 → src[1*2+0]=3.0
        assert_eq!(output[0], 3.0);
        assert_eq!(output[1], 2.0);
    }

    #[test]
    fn test_gather_single_element() {
        let src = [42.0];
        let indices = [0];
        let kernel = ScatterGatherKernel::new(1, 1).unwrap();
        let config = GatherConfig::new(0, (1, 1), true).unwrap();
        let mut output = [0.0_f32; 1];
        gather_cpu(&src, &indices, &mut output, &kernel, &config).unwrap();
        assert_eq!(output[0], 42.0);
    }

    #[test]
    fn test_gather_shape_mismatch_axis0() {
        let src = [1.0; 6]; // 3×2
        let indices = [0, 1, 2]; // 1×3, but src has 2 cols
        let kernel = ScatterGatherKernel::new(3, 2).unwrap();
        let config = GatherConfig::new(0, (1, 3), true).unwrap();
        let mut output = [0.0_f32; 3];
        assert!(gather_cpu(&src, &indices, &mut output, &kernel, &config).is_err());
    }

    #[test]
    fn test_gather_large_tensor() {
        let rows = 100;
        let cols = 64;
        let src: Vec<f32> = (0..(rows * cols) as u32).map(|x| x as f32).collect();
        let n_sel = 50;
        let indices: Vec<usize> = (0..n_sel).flat_map(|i| vec![i * 2; cols]).collect();
        let kernel = ScatterGatherKernel::new(rows, cols).unwrap();
        let config = GatherConfig::new(0, (n_sel, cols), true).unwrap();
        let mut output = vec![0.0_f32; n_sel * cols];
        gather_cpu(&src, &indices, &mut output, &kernel, &config).unwrap();
        for j in 0..cols {
            assert_eq!(output[j], src[j]);
        }
        for j in 0..cols {
            assert_eq!(output[cols + j], src[2 * cols + j]);
        }
    }

    // -- Scatter CPU tests --------------------------------------------------

    #[test]
    fn test_scatter_assign_axis0() {
        let src = [10.0, 11.0];
        let indices = [2, 2]; // put into row 2
        let config = GatherConfig::new(0, (1, 2), true).unwrap();
        let mut dst = [0.0_f32; 6];
        scatter_cpu(&src, &indices, &mut dst, (3, 2), &config, ScatterMode::Assign).unwrap();
        assert_eq!(dst, [0.0, 0.0, 0.0, 0.0, 10.0, 11.0]);
    }

    #[test]
    fn test_scatter_add_accumulates() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 0, 0, 0]; // both rows target row 0
        let config = GatherConfig::new(0, (2, 2), true).unwrap();
        let mut dst = [0.0_f32; 4];
        scatter_cpu(&src, &indices, &mut dst, (2, 2), &config, ScatterMode::Add).unwrap();
        assert_eq!(dst[0], 4.0);
        assert_eq!(dst[1], 6.0);
    }

    #[test]
    fn test_scatter_max_keeps_max() {
        let src = [5.0, 1.0, 3.0, 9.0];
        let indices = [0, 0, 0, 0];
        let config = GatherConfig::new(0, (2, 2), true).unwrap();
        let mut dst = [0.0_f32; 4];
        scatter_cpu(&src, &indices, &mut dst, (2, 2), &config, ScatterMode::Max).unwrap();
        assert_eq!(dst[0], 5.0);
        assert_eq!(dst[1], 9.0);
    }

    #[test]
    fn test_scatter_min_keeps_min() {
        let src = [5.0, 1.0, 3.0, 9.0];
        let indices = [0, 0, 0, 0];
        let config = GatherConfig::new(0, (2, 2), true).unwrap();
        let mut dst = [f32::INFINITY; 4];
        scatter_cpu(&src, &indices, &mut dst, (2, 2), &config, ScatterMode::Min).unwrap();
        assert_eq!(dst[0], 3.0);
        assert_eq!(dst[1], 1.0);
    }

    #[test]
    fn test_scatter_out_of_bounds_error() {
        let src = [1.0, 2.0];
        let indices = [99, 0]; // OOB
        let config = GatherConfig::new(0, (1, 2), true).unwrap();
        let mut dst = [0.0_f32; 4];
        assert!(
            scatter_cpu(&src, &indices, &mut dst, (2, 2), &config, ScatterMode::Assign).is_err()
        );
    }

    #[test]
    fn test_scatter_axis1() {
        let src = [10.0, 20.0]; // 2×1
        let indices = [2, 0]; // row0→col2, row1→col0
        let config = GatherConfig::new(1, (2, 1), true).unwrap();
        let mut dst = [0.0_f32; 6];
        scatter_cpu(&src, &indices, &mut dst, (2, 3), &config, ScatterMode::Assign).unwrap();
        assert_eq!(dst, [0.0, 0.0, 10.0, 20.0, 0.0, 0.0]);
    }

    // -- Index select tests -------------------------------------------------

    #[test]
    fn test_index_select_basic() {
        let src: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let indices = [3, 1]; // select rows 3 and 1
        let kernel = ScatterGatherKernel::new(4, 3).unwrap();
        let mut output = [0.0_f32; 6];
        index_select_cpu(&src, &indices, &mut output, &kernel, true).unwrap();
        assert_eq!(output, [9.0, 10.0, 11.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_index_select_single_row() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0];
        let kernel = ScatterGatherKernel::new(1, 3).unwrap();
        let mut output = [0.0_f32; 3];
        index_select_cpu(&src, &indices, &mut output, &kernel, true).unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_index_select_duplicate_indices() {
        let src = [10.0, 20.0, 30.0, 40.0];
        let indices = [1, 1, 0]; // row 1 twice, row 0 once
        let kernel = ScatterGatherKernel::new(2, 2).unwrap();
        let mut output = [0.0_f32; 6];
        index_select_cpu(&src, &indices, &mut output, &kernel, true).unwrap();
        assert_eq!(output, [30.0, 40.0, 30.0, 40.0, 10.0, 20.0]);
    }

    #[test]
    fn test_index_select_out_of_bounds() {
        let src = [1.0, 2.0];
        let indices = [5];
        let kernel = ScatterGatherKernel::new(1, 2).unwrap();
        let mut output = [0.0_f32; 2];
        assert!(index_select_cpu(&src, &indices, &mut output, &kernel, true).is_err());
    }

    #[test]
    fn test_index_select_empty_indices() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices: &[usize] = &[];
        let kernel = ScatterGatherKernel::new(2, 2).unwrap();
        let mut output: Vec<f32> = vec![];
        index_select_cpu(&src, indices, &mut output, &kernel, true).unwrap();
    }

    // -- Unified dispatch tests ---------------------------------------------

    #[test]
    fn test_gather_forward_dispatches_cpu() {
        let src = [10.0, 11.0, 20.0, 21.0];
        let indices = [1, 0];
        let kernel = ScatterGatherKernel::new(2, 2).unwrap();
        let config = GatherConfig::new(0, (1, 2), true).unwrap();
        let mut output = [0.0_f32; 2];
        gather_forward(&src, &indices, &mut output, &kernel, &config).unwrap();
        assert_eq!(output, [20.0, 11.0]);
    }

    #[test]
    fn test_scatter_forward_dispatches_cpu() {
        let src = [7.0, 8.0];
        let indices = [1, 1];
        let config = GatherConfig::new(0, (1, 2), true).unwrap();
        let mut dst = [0.0_f32; 4];
        scatter_forward(&src, &indices, &mut dst, (2, 2), &config, ScatterMode::Assign).unwrap();
        assert_eq!(dst, [0.0, 0.0, 7.0, 8.0]);
    }

    #[test]
    fn test_gather_forward_matches_cpu() {
        let src: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let indices = [3, 1, 0, 2, 3]; // 1×5, all within [0,4)
        let kernel = ScatterGatherKernel::new(4, 5).unwrap();
        let config = GatherConfig::new(0, (1, 5), true).unwrap();

        let mut out_fwd = [0.0_f32; 5];
        let mut out_cpu = [0.0_f32; 5];

        gather_forward(&src, &indices, &mut out_fwd, &kernel, &config).unwrap();
        gather_cpu(&src, &indices, &mut out_cpu, &kernel, &config).unwrap();

        assert_eq!(out_fwd, out_cpu);
    }

    // -- GPU launch stub tests ----------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gather_launch() {
        let src = vec![1.0_f32; 1024];
        let indices = vec![0_usize; 256];
        let kernel = ScatterGatherKernel::new(4, 256).unwrap();
        let config = GatherConfig::new(0, (1, 256), true).unwrap();
        let mut output = vec![0.0_f32; 256];
        let result = gather_forward(&src, &indices, &mut output, &kernel, &config);
        assert!(result.is_ok(), "CUDA gather launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_scatter_launch() {
        let src = vec![1.0_f32; 256];
        let indices = vec![0_usize; 256];
        let config = GatherConfig::new(0, (1, 256), true).unwrap();
        let mut dst = vec![0.0_f32; 1024];
        let result = scatter_forward(&src, &indices, &mut dst, (4, 256), &config, ScatterMode::Add);
        assert!(result.is_ok(), "CUDA scatter launch failed: {result:?}");
    }
}
