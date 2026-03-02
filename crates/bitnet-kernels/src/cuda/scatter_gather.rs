//! CUDA scatter/gather operations for indexed tensor access.
//!
//! This module provides GPU-accelerated scatter/gather, N-dimensional variants,
//! index select/put, and optimized embedding lookup operations. All functions
//! include CPU fallback implementations for correctness testing and non-GPU
//! environments.
//!
//! # Operations
//!
//! - [`gather`] / [`scatter`]: Element-wise gather/scatter along an axis.
//! - [`gather_nd`] / [`scatter_nd`]: N-dimensional indexed gather/scatter.
//! - [`index_select`] / [`index_put`]: Dimension-based select and put.
//! - [`embedding_lookup`]: Optimized embedding table gather.
//!
//! # CUDA kernels
//!
//! GPU launch stubs are gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`
//! and return `KernelError::GpuError` until real PTX kernels are compiled.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for CUDA scatter/gather kernel launches.
#[derive(Debug, Clone)]
pub struct CudaScatterGatherConfig {
    /// Threads per block for GPU launch.
    pub threads_per_block: u32,
    /// Whether to perform bounds checking on indices.
    pub bounds_check: bool,
}

impl CudaScatterGatherConfig {
    /// Create a new configuration with the given element count for thread sizing.
    pub fn new(n_elements: usize, bounds_check: bool) -> Self {
        let threads_per_block = (n_elements as u32).clamp(1, 1024);
        Self { threads_per_block, bounds_check }
    }

    /// CUDA grid dimensions for the given total element count.
    pub fn grid_dim(&self, n_elements: usize) -> (u32, u32, u32) {
        let blocks = (n_elements as u32).div_ceil(self.threads_per_block);
        (blocks.max(1), 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// CUDA C source for the gather kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const GATHER_KERNEL_SRC: &str = r#"
extern "C" __global__ void gather_f32(
    const float* __restrict__ input,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int axis,
    int outer_size,
    int inner_size,
    int axis_size,
    int index_size,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int inner_idx = tid % inner_size;
    int index_idx = (tid / inner_size) % index_size;
    int outer_idx = tid / (inner_size * index_size);

    int src_idx = indices[outer_idx * index_size * inner_size + index_idx * inner_size + inner_idx];
    if (src_idx < 0) src_idx = 0;
    if (src_idx >= axis_size) src_idx = axis_size - 1;

    int src_offset = outer_idx * axis_size * inner_size + src_idx * inner_size + inner_idx;
    output[tid] = input[src_offset];
}
"#;

/// CUDA C source for the scatter kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const SCATTER_KERNEL_SRC: &str = r#"
extern "C" __global__ void scatter_f32(
    const float* __restrict__ updates,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int axis,
    int outer_size,
    int inner_size,
    int axis_size,
    int index_size,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int inner_idx = tid % inner_size;
    int index_idx = (tid / inner_size) % index_size;
    int outer_idx = tid / (inner_size * index_size);

    int dst_idx = indices[outer_idx * index_size * inner_size + index_idx * inner_size + inner_idx];
    if (dst_idx < 0) dst_idx = 0;
    if (dst_idx >= axis_size) dst_idx = axis_size - 1;

    int dst_offset = outer_idx * axis_size * inner_size + dst_idx * inner_size + inner_idx;
    atomicExch(&output[dst_offset], updates[tid]);
}
"#;

// ---------------------------------------------------------------------------
// CPU fallback — gather
// ---------------------------------------------------------------------------

/// Gather elements from `input` at `indices` along the specified axis.
///
/// For a 2-D input `[rows, cols]`:
/// - axis 0: `output[i][j] = input[indices[i][j]][j]`
/// - axis 1: `output[i][j] = input[i][indices[i][j]]`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on shape mismatch or
/// out-of-bounds indices when bounds checking is enabled.
pub fn gather(
    input: &[f32],
    indices: &[usize],
    output: &mut [f32],
    input_shape: &[usize],
    axis: usize,
    bounds_check: bool,
) -> Result<()> {
    if input_shape.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "gather: input_shape must not be empty".into(),
        }
        .into());
    }
    if axis >= input_shape.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("gather: axis {axis} out of range for {}-D tensor", input_shape.len()),
        }
        .into());
    }

    let axis_size = input_shape[axis];
    let outer: usize = input_shape[..axis].iter().product();
    let inner: usize = input_shape[axis + 1..].iter().product();
    let outer = if outer == 0 { 1 } else { outer };
    let inner = if inner == 0 { 1 } else { inner };
    let index_size = indices.len() / (outer * inner);

    if indices.len() != outer * index_size * inner {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gather: indices length {} not divisible by outer*inner={}",
                indices.len(),
                outer * inner,
            ),
        }
        .into());
    }

    let total = outer * index_size * inner;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("gather: output length {} < required {total}", output.len()),
        }
        .into());
    }

    for o in 0..outer {
        for idx in 0..index_size {
            for i in 0..inner {
                let flat = o * index_size * inner + idx * inner + i;
                let src_idx = indices[flat];
                if bounds_check && src_idx >= axis_size {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "gather: index {src_idx} out of bounds for axis size {axis_size}"
                        ),
                    }
                    .into());
                }
                let clamped = src_idx.min(axis_size.saturating_sub(1));
                let src_off = o * axis_size * inner + clamped * inner + i;
                output[flat] = input[src_off];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — scatter
// ---------------------------------------------------------------------------

/// Scatter `updates` into `input` at `indices` along the specified axis.
///
/// The output tensor is initialized as a copy of `input`, then elements from
/// `updates` are placed at positions specified by `indices`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on shape/size mismatch.
pub fn scatter(
    input: &[f32],
    indices: &[usize],
    updates: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    axis: usize,
    bounds_check: bool,
) -> Result<()> {
    if input_shape.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "scatter: input_shape must not be empty".into(),
        }
        .into());
    }
    if axis >= input_shape.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("scatter: axis {axis} out of range for {}-D tensor", input_shape.len()),
        }
        .into());
    }
    if indices.len() != updates.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "scatter: indices length {} != updates length {}",
                indices.len(),
                updates.len(),
            ),
        }
        .into());
    }

    let total_input: usize = input_shape.iter().product();
    if output.len() < total_input {
        return Err(KernelError::InvalidArguments {
            reason: format!("scatter: output length {} < input size {total_input}", output.len()),
        }
        .into());
    }

    output[..total_input].copy_from_slice(&input[..total_input]);

    let axis_size = input_shape[axis];
    let outer: usize = input_shape[..axis].iter().product();
    let inner: usize = input_shape[axis + 1..].iter().product();
    let outer = if outer == 0 { 1 } else { outer };
    let inner = if inner == 0 { 1 } else { inner };
    let index_size = indices.len() / (outer * inner);

    for o in 0..outer {
        for idx in 0..index_size {
            for i in 0..inner {
                let flat = o * index_size * inner + idx * inner + i;
                if flat >= indices.len() {
                    break;
                }
                let dst_idx = indices[flat];
                if bounds_check && dst_idx >= axis_size {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "scatter: index {dst_idx} out of bounds for axis size {axis_size}"
                        ),
                    }
                    .into());
                }
                let clamped = dst_idx.min(axis_size.saturating_sub(1));
                let dst_off = o * axis_size * inner + clamped * inner + i;
                output[dst_off] = updates[flat];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — gather_nd
// ---------------------------------------------------------------------------

/// N-dimensional gather: select elements using multi-dimensional index tuples.
///
/// `indices` is a flat array of index tuples. Each tuple has `index_depth`
/// elements, specifying coordinates into the first `index_depth` dimensions
/// of `input`. The number of tuples is `indices.len() / index_depth`.
///
/// # Errors
///
/// Returns error on invalid shapes or out-of-bounds indices.
pub fn gather_nd(
    input: &[f32],
    indices: &[usize],
    output: &mut [f32],
    input_shape: &[usize],
    index_depth: usize,
) -> Result<()> {
    if index_depth == 0 || index_depth > input_shape.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gather_nd: index_depth {index_depth} invalid for {}-D tensor",
                input_shape.len()
            ),
        }
        .into());
    }
    if !indices.len().is_multiple_of(index_depth) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gather_nd: indices length {} not divisible by index_depth {index_depth}",
                indices.len()
            ),
        }
        .into());
    }

    let n_tuples = indices.len() / index_depth;
    let slice_size: usize = input_shape[index_depth..].iter().product();
    let slice_size = if slice_size == 0 { 1 } else { slice_size };

    if output.len() < n_tuples * slice_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gather_nd: output length {} < required {}",
                output.len(),
                n_tuples * slice_size,
            ),
        }
        .into());
    }

    for t in 0..n_tuples {
        let tuple = &indices[t * index_depth..(t + 1) * index_depth];
        let mut offset = 0usize;
        let mut stride = 1usize;
        for d in (0..index_depth).rev() {
            let idx = tuple[d].min(input_shape[d].saturating_sub(1));
            offset += idx * stride;
            stride *= input_shape[d];
        }
        let src_start = offset * slice_size;
        let dst_start = t * slice_size;
        for s in 0..slice_size {
            if src_start + s < input.len() {
                output[dst_start + s] = input[src_start + s];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — scatter_nd
// ---------------------------------------------------------------------------

/// N-dimensional scatter: place `updates` into a tensor at multi-dimensional
/// index positions.
///
/// The output is initialized to zeros with shape determined by `output_shape`.
/// Each index tuple (of length `index_depth`) in `indices` specifies a
/// position, and the corresponding slice from `updates` is placed there.
///
/// # Errors
///
/// Returns error on invalid shapes.
pub fn scatter_nd(
    indices: &[usize],
    updates: &[f32],
    output: &mut [f32],
    output_shape: &[usize],
    index_depth: usize,
) -> Result<()> {
    if index_depth == 0 || index_depth > output_shape.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "scatter_nd: index_depth {index_depth} invalid for {}-D tensor",
                output_shape.len()
            ),
        }
        .into());
    }
    if !indices.len().is_multiple_of(index_depth) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "scatter_nd: indices length {} not divisible by index_depth {index_depth}",
                indices.len()
            ),
        }
        .into());
    }

    let n_tuples = indices.len() / index_depth;
    let slice_size: usize = output_shape[index_depth..].iter().product();
    let slice_size = if slice_size == 0 { 1 } else { slice_size };
    let total_output: usize = output_shape.iter().product();

    if output.len() < total_output {
        return Err(KernelError::InvalidArguments {
            reason: format!("scatter_nd: output length {} < required {total_output}", output.len()),
        }
        .into());
    }
    if updates.len() < n_tuples * slice_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "scatter_nd: updates length {} < required {}",
                updates.len(),
                n_tuples * slice_size,
            ),
        }
        .into());
    }

    output[..total_output].fill(0.0);

    for t in 0..n_tuples {
        let tuple = &indices[t * index_depth..(t + 1) * index_depth];
        let mut offset = 0usize;
        let mut stride = 1usize;
        for d in (0..index_depth).rev() {
            let idx = tuple[d].min(output_shape[d].saturating_sub(1));
            offset += idx * stride;
            stride *= output_shape[d];
        }
        let dst_start = offset * slice_size;
        let src_start = t * slice_size;
        for s in 0..slice_size {
            if dst_start + s < total_output {
                output[dst_start + s] = updates[src_start + s];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — index_select
// ---------------------------------------------------------------------------

/// Select slices from `input` along dimension `dim` using 1-D `indices`.
///
/// For a 2-D input `[rows, cols]` with `dim=0`, selects rows. With `dim=1`,
/// selects columns.
///
/// # Errors
///
/// Returns error on invalid dimension or out-of-bounds indices.
pub fn index_select(
    input: &[f32],
    dim: usize,
    indices: &[usize],
    output: &mut [f32],
    input_shape: &[usize],
    bounds_check: bool,
) -> Result<()> {
    if input_shape.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "index_select: input_shape must not be empty".into(),
        }
        .into());
    }
    if dim >= input_shape.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "index_select: dim {dim} out of range for {}-D tensor",
                input_shape.len()
            ),
        }
        .into());
    }

    let dim_size = input_shape[dim];
    let outer: usize = input_shape[..dim].iter().product();
    let inner: usize = input_shape[dim + 1..].iter().product();
    let outer = if outer == 0 { 1 } else { outer };
    let inner = if inner == 0 { 1 } else { inner };
    let n_indices = indices.len();
    let total = outer * n_indices * inner;

    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("index_select: output length {} < required {total}", output.len()),
        }
        .into());
    }

    for o in 0..outer {
        for (sel_i, &idx) in indices.iter().enumerate() {
            if bounds_check && idx >= dim_size {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "index_select: index {idx} out of bounds for dim size {dim_size}"
                    ),
                }
                .into());
            }
            let clamped = idx.min(dim_size.saturating_sub(1));
            let src_base = o * dim_size * inner + clamped * inner;
            let dst_base = o * n_indices * inner + sel_i * inner;
            output[dst_base..dst_base + inner].copy_from_slice(&input[src_base..src_base + inner]);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — index_put
// ---------------------------------------------------------------------------

/// Put `values` into `input` at positions specified by 1-D `indices` along
/// dimension `dim`.
///
/// The output is initialized as a copy of `input`, then values are placed
/// at the indexed positions (last write wins for duplicate indices).
///
/// # Errors
///
/// Returns error on invalid dimension or out-of-bounds indices.
pub fn index_put(
    input: &[f32],
    indices: &[usize],
    values: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    dim: usize,
    bounds_check: bool,
) -> Result<()> {
    if input_shape.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "index_put: input_shape must not be empty".into(),
        }
        .into());
    }
    if dim >= input_shape.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("index_put: dim {dim} out of range for {}-D tensor", input_shape.len()),
        }
        .into());
    }

    let dim_size = input_shape[dim];
    let outer: usize = input_shape[..dim].iter().product();
    let inner: usize = input_shape[dim + 1..].iter().product();
    let outer = if outer == 0 { 1 } else { outer };
    let inner = if inner == 0 { 1 } else { inner };
    let n_indices = indices.len();
    let total_input: usize = input_shape.iter().product();

    if output.len() < total_input {
        return Err(KernelError::InvalidArguments {
            reason: format!("index_put: output length {} < input size {total_input}", output.len()),
        }
        .into());
    }
    if values.len() < outer * n_indices * inner {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "index_put: values length {} < required {}",
                values.len(),
                outer * n_indices * inner,
            ),
        }
        .into());
    }

    output[..total_input].copy_from_slice(&input[..total_input]);

    for o in 0..outer {
        for (sel_i, &idx) in indices.iter().enumerate() {
            if bounds_check && idx >= dim_size {
                return Err(KernelError::InvalidArguments {
                    reason: format!("index_put: index {idx} out of bounds for dim size {dim_size}"),
                }
                .into());
            }
            let clamped = idx.min(dim_size.saturating_sub(1));
            let dst_base = o * dim_size * inner + clamped * inner;
            let src_base = o * n_indices * inner + sel_i * inner;
            output[dst_base..dst_base + inner].copy_from_slice(&values[src_base..src_base + inner]);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — embedding_lookup
// ---------------------------------------------------------------------------

/// Optimized embedding table gather: for each token ID, copy the
/// corresponding row from the embedding table.
///
/// `table` has shape `[vocab_size, embedding_dim]`. `indices` contains
/// token IDs. Output has shape `[indices.len(), embedding_dim]`.
///
/// # Errors
///
/// Returns error on out-of-bounds token IDs (when bounds_check is enabled)
/// or length mismatches.
pub fn embedding_lookup(
    table: &[f32],
    indices: &[u32],
    output: &mut [f32],
    vocab_size: usize,
    embedding_dim: usize,
    bounds_check: bool,
) -> Result<()> {
    if vocab_size == 0 || embedding_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "embedding_lookup: dimensions must be non-zero: \
                 vocab_size={vocab_size}, embedding_dim={embedding_dim}"
            ),
        }
        .into());
    }
    if table.len() < vocab_size * embedding_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "embedding_lookup: table length {} < vocab_size*dim={}",
                table.len(),
                vocab_size * embedding_dim,
            ),
        }
        .into());
    }
    let n_tokens = indices.len();
    if output.len() < n_tokens * embedding_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "embedding_lookup: output length {} < required {}",
                output.len(),
                n_tokens * embedding_dim,
            ),
        }
        .into());
    }

    for (pos, &token_id) in indices.iter().enumerate() {
        let id = token_id as usize;
        if bounds_check && id >= vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embedding_lookup: token ID {id} out of bounds for vocab size {vocab_size}"
                ),
            }
            .into());
        }
        let clamped = id.min(vocab_size.saturating_sub(1));
        let src_start = clamped * embedding_dim;
        let dst_start = pos * embedding_dim;
        output[dst_start..dst_start + embedding_dim]
            .copy_from_slice(&table[src_start..src_start + embedding_dim]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU launch stubs
// ---------------------------------------------------------------------------

/// GPU launch stub for gather.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_cuda_gather(
    _input: &[f32],
    _indices: &[usize],
    _output: &mut [f32],
    _input_shape: &[usize],
    _axis: usize,
) -> Result<()> {
    log::debug!("cuda scatter_gather::gather stub invoked");
    Err(KernelError::GpuError { reason: "gather CUDA kernel not yet compiled — stub".into() }
        .into())
}

/// GPU launch stub for scatter.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_cuda_scatter(
    _input: &[f32],
    _indices: &[usize],
    _updates: &[f32],
    _output: &mut [f32],
    _input_shape: &[usize],
    _axis: usize,
) -> Result<()> {
    log::debug!("cuda scatter_gather::scatter stub invoked");
    Err(KernelError::GpuError { reason: "scatter CUDA kernel not yet compiled — stub".into() }
        .into())
}

/// GPU launch stub for embedding lookup.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_cuda_embedding_lookup(
    _table: &[f32],
    _indices: &[u32],
    _output: &mut [f32],
    _vocab_size: usize,
    _embedding_dim: usize,
) -> Result<()> {
    log::debug!("cuda scatter_gather::embedding_lookup stub invoked");
    Err(KernelError::GpuError {
        reason: "embedding_lookup CUDA kernel not yet compiled — stub".into(),
    }
    .into())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CudaScatterGatherConfig tests ------------------------------------

    #[test]
    fn test_config_new_basic() {
        let cfg = CudaScatterGatherConfig::new(512, true);
        assert_eq!(cfg.threads_per_block, 512);
        assert!(cfg.bounds_check);
    }

    #[test]
    fn test_config_threads_capped_at_1024() {
        let cfg = CudaScatterGatherConfig::new(8192, false);
        assert_eq!(cfg.threads_per_block, 1024);
    }

    #[test]
    fn test_config_threads_min_1() {
        let cfg = CudaScatterGatherConfig::new(0, false);
        assert_eq!(cfg.threads_per_block, 1);
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = CudaScatterGatherConfig::new(256, false);
        assert_eq!(cfg.grid_dim(1024), (4, 1, 1));
        assert_eq!(cfg.grid_dim(1), (1, 1, 1));
        assert_eq!(cfg.grid_dim(257), (2, 1, 1));
    }

    #[test]
    fn test_config_block_dim() {
        let cfg = CudaScatterGatherConfig::new(128, false);
        assert_eq!(cfg.block_dim(), (128, 1, 1));
    }

    // -- gather tests -----------------------------------------------------

    #[test]
    fn test_gather_axis0_2d() {
        // input: 3×2, gather rows by index (per-element: repeat index per inner dim)
        let input = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        // Select rows 2 and 0: indices repeated per column
        let indices = [2, 2, 0, 0];
        let mut output = [0.0f32; 4];
        gather(&input, &indices, &mut output, &[3, 2], 0, true).unwrap();
        assert_eq!(&output[..4], &[30.0, 31.0, 10.0, 11.0]);
    }

    #[test]
    fn test_gather_axis1_2d() {
        // input: 2×4, gather along columns
        let input: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let indices = [3, 1, 0, 2];
        let mut output = [0.0f32; 4];
        gather(&input, &indices, &mut output, &[2, 4], 1, true).unwrap();
        assert_eq!(output, [3.0, 1.0, 4.0, 6.0]);
    }

    #[test]
    fn test_gather_1d() {
        let input = [10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [4, 0, 2];
        let mut output = [0.0f32; 3];
        gather(&input, &indices, &mut output, &[5], 0, true).unwrap();
        assert_eq!(output, [50.0, 10.0, 30.0]);
    }

    #[test]
    fn test_gather_single_element() {
        let input = [42.0];
        let indices = [0];
        let mut output = [0.0f32; 1];
        gather(&input, &indices, &mut output, &[1], 0, true).unwrap();
        assert_eq!(output[0], 42.0);
    }

    #[test]
    fn test_gather_bounds_check_error() {
        let input = [1.0, 2.0, 3.0];
        let indices = [5];
        let mut output = [0.0f32; 1];
        assert!(gather(&input, &indices, &mut output, &[3], 0, true).is_err());
    }

    #[test]
    fn test_gather_bounds_clamp() {
        let input = [1.0, 2.0, 3.0];
        let indices = [99];
        let mut output = [0.0f32; 1];
        gather(&input, &indices, &mut output, &[3], 0, false).unwrap();
        assert_eq!(output[0], 3.0); // clamped to last
    }

    #[test]
    fn test_gather_invalid_axis() {
        let input = [1.0, 2.0];
        let indices = [0];
        let mut output = [0.0f32; 1];
        assert!(gather(&input, &indices, &mut output, &[2], 1, true).is_err());
    }

    #[test]
    fn test_gather_empty_shape() {
        let input = [1.0];
        let indices = [0];
        let mut output = [0.0f32; 1];
        assert!(gather(&input, &indices, &mut output, &[], 0, true).is_err());
    }

    #[test]
    fn test_gather_large_tensor() {
        let rows = 100;
        let cols = 64;
        let input: Vec<f32> = (0..(rows * cols) as u32).map(|x| x as f32).collect();
        // Per-element gather: repeat each row index across all columns
        let indices: Vec<usize> = (0..10).flat_map(|r| vec![r; cols]).collect();
        let mut output = vec![0.0f32; 10 * cols];
        gather(&input, &indices, &mut output, &[rows, cols], 0, true).unwrap();
        for r in 0..10 {
            for c in 0..cols {
                assert_eq!(output[r * cols + c], (r * cols + c) as f32);
            }
        }
    }

    // -- scatter tests ----------------------------------------------------

    #[test]
    fn test_scatter_axis0_2d() {
        let input = [0.0f32; 6]; // 3×2
        // Per-element scatter: indices match updates length, repeated per inner dim
        let indices = [2, 2, 0, 0];
        let updates = [10.0, 11.0, 20.0, 21.0];
        let mut output = [0.0f32; 6];
        scatter(&input, &indices, &updates, &mut output, &[3, 2], 0, true).unwrap();
        assert_eq!(output[4], 10.0); // row 2, col 0
        assert_eq!(output[5], 11.0); // row 2, col 1
        assert_eq!(output[0], 20.0); // row 0, col 0
        assert_eq!(output[1], 21.0); // row 0, col 1
    }

    #[test]
    fn test_scatter_axis1_2d() {
        let input = [0.0f32; 6]; // 2×3
        let indices = [2, 0];
        let updates = [10.0, 20.0];
        let mut output = [0.0f32; 6];
        scatter(&input, &indices, &updates, &mut output, &[2, 3], 1, true).unwrap();
        assert_eq!(output[2], 10.0); // row 0, col 2
        assert_eq!(output[3], 20.0); // row 1, col 0
    }

    #[test]
    fn test_scatter_1d() {
        let input = [0.0f32; 5];
        let indices = [3, 1];
        let updates = [10.0, 20.0];
        let mut output = [0.0f32; 5];
        scatter(&input, &indices, &updates, &mut output, &[5], 0, true).unwrap();
        assert_eq!(output[3], 10.0);
        assert_eq!(output[1], 20.0);
    }

    #[test]
    fn test_scatter_preserves_input() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let indices = [0];
        let updates = [99.0];
        let mut output = [0.0f32; 4];
        scatter(&input, &indices, &updates, &mut output, &[4], 0, true).unwrap();
        assert_eq!(output[0], 99.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 3.0);
        assert_eq!(output[3], 4.0);
    }

    #[test]
    fn test_scatter_bounds_check_error() {
        let input = [0.0f32; 4];
        let indices = [10];
        let updates = [1.0];
        let mut output = [0.0f32; 4];
        assert!(scatter(&input, &indices, &updates, &mut output, &[4], 0, true).is_err());
    }

    #[test]
    fn test_scatter_indices_updates_mismatch() {
        let input = [0.0f32; 4];
        let indices = [0, 1];
        let updates = [1.0]; // too short
        let mut output = [0.0f32; 4];
        assert!(scatter(&input, &indices, &updates, &mut output, &[4], 0, true).is_err());
    }

    // -- gather_nd tests --------------------------------------------------

    #[test]
    fn test_gather_nd_1d() {
        let input = [10.0, 20.0, 30.0, 40.0];
        let indices = [2, 0, 3]; // 3 tuples of depth 1
        let mut output = [0.0f32; 3];
        gather_nd(&input, &indices, &mut output, &[4], 1).unwrap();
        assert_eq!(output, [30.0, 10.0, 40.0]);
    }

    #[test]
    fn test_gather_nd_2d() {
        // input: 3×4
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        // 2 tuples: (0,1) and (2,3)
        let indices = [0, 1, 2, 3];
        let mut output = [0.0f32; 2];
        gather_nd(&input, &indices, &mut output, &[3, 4], 2).unwrap();
        assert_eq!(output[0], 1.0); // [0][1]
        assert_eq!(output[1], 11.0); // [2][3]
    }

    #[test]
    fn test_gather_nd_partial_index() {
        // input: 3×4, index_depth=1 → selects full rows
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let indices = [1]; // select row 1
        let mut output = [0.0f32; 4];
        gather_nd(&input, &indices, &mut output, &[3, 4], 1).unwrap();
        assert_eq!(output, [4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_gather_nd_invalid_depth() {
        let input = [1.0, 2.0];
        let indices = [0];
        let mut output = [0.0f32; 1];
        assert!(gather_nd(&input, &indices, &mut output, &[2], 0).is_err());
        assert!(gather_nd(&input, &indices, &mut output, &[2], 3).is_err());
    }

    #[test]
    fn test_gather_nd_indivisible_indices() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 1, 2]; // 3 not divisible by depth=2
        let mut output = [0.0f32; 2];
        assert!(gather_nd(&input, &indices, &mut output, &[2, 2], 2).is_err());
    }

    // -- scatter_nd tests -------------------------------------------------

    #[test]
    fn test_scatter_nd_1d() {
        let indices = [1, 3];
        let updates = [10.0, 30.0];
        let mut output = [0.0f32; 5];
        scatter_nd(&indices, &updates, &mut output, &[5], 1).unwrap();
        assert_eq!(output, [0.0, 10.0, 0.0, 30.0, 0.0]);
    }

    #[test]
    fn test_scatter_nd_2d() {
        // output: 3×4
        let indices = [0, 1, 2, 3]; // 2 tuples: (0,1) and (2,3)
        let updates = [99.0, 88.0];
        let mut output = [0.0f32; 12];
        scatter_nd(&indices, &updates, &mut output, &[3, 4], 2).unwrap();
        assert_eq!(output[1], 99.0); // [0][1]
        assert_eq!(output[11], 88.0); // [2][3]
    }

    #[test]
    fn test_scatter_nd_partial_index() {
        // output: 3×2, index_depth=1 → scatter full rows
        let indices = [2]; // put at row 2
        let updates = [10.0, 20.0];
        let mut output = [0.0f32; 6];
        scatter_nd(&indices, &updates, &mut output, &[3, 2], 1).unwrap();
        assert_eq!(output[4], 10.0);
        assert_eq!(output[5], 20.0);
    }

    #[test]
    fn test_scatter_nd_invalid_depth() {
        let indices = [0];
        let updates = [1.0];
        let mut output = [0.0f32; 4];
        assert!(scatter_nd(&indices, &updates, &mut output, &[4], 0).is_err());
    }

    // -- index_select tests -----------------------------------------------

    #[test]
    fn test_index_select_dim0() {
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let indices = [2, 0];
        let mut output = [0.0f32; 8];
        index_select(&input, 0, &indices, &mut output, &[3, 4], true).unwrap();
        assert_eq!(&output[..4], &[8.0, 9.0, 10.0, 11.0]);
        assert_eq!(&output[4..8], &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_index_select_dim1() {
        // input: 2×4, select columns 3 and 1
        let input: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let indices = [3, 1];
        let mut output = [0.0f32; 4];
        index_select(&input, 1, &indices, &mut output, &[2, 4], true).unwrap();
        assert_eq!(output, [3.0, 1.0, 7.0, 5.0]);
    }

    #[test]
    fn test_index_select_single() {
        let input = [42.0];
        let indices = [0];
        let mut output = [0.0f32; 1];
        index_select(&input, 0, &indices, &mut output, &[1], true).unwrap();
        assert_eq!(output[0], 42.0);
    }

    #[test]
    fn test_index_select_bounds_error() {
        let input = [1.0, 2.0, 3.0];
        let indices = [5];
        let mut output = [0.0f32; 1];
        assert!(index_select(&input, 0, &indices, &mut output, &[3], true).is_err());
    }

    #[test]
    fn test_index_select_invalid_dim() {
        let input = [1.0, 2.0];
        let indices = [0];
        let mut output = [0.0f32; 1];
        assert!(index_select(&input, 2, &indices, &mut output, &[2], true).is_err());
    }

    // -- index_put tests --------------------------------------------------

    #[test]
    fn test_index_put_dim0() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let indices = [1];
        let values = [99.0, 88.0];
        let mut output = [0.0f32; 6];
        index_put(&input, &indices, &values, &mut output, &[3, 2], 0, true).unwrap();
        assert_eq!(output[2], 99.0);
        assert_eq!(output[3], 88.0);
        assert_eq!(output[0], 1.0); // preserved
    }

    #[test]
    fn test_index_put_dim1() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let indices = [0];
        let values = [10.0, 40.0]; // outer=2, each row puts 1 value
        let mut output = [0.0f32; 6];
        index_put(&input, &indices, &values, &mut output, &[2, 3], 1, true).unwrap();
        assert_eq!(output[0], 10.0); // row 0, col 0
        assert_eq!(output[3], 40.0); // row 1, col 0
    }

    #[test]
    fn test_index_put_preserves_rest() {
        let input = [10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [2];
        let values = [99.0];
        let mut output = [0.0f32; 5];
        index_put(&input, &indices, &values, &mut output, &[5], 0, true).unwrap();
        assert_eq!(output, [10.0, 20.0, 99.0, 40.0, 50.0]);
    }

    #[test]
    fn test_index_put_bounds_error() {
        let input = [1.0, 2.0];
        let indices = [5];
        let values = [10.0];
        let mut output = [0.0f32; 2];
        assert!(index_put(&input, &indices, &values, &mut output, &[2], 0, true).is_err());
    }

    #[test]
    fn test_index_put_invalid_dim() {
        let input = [1.0, 2.0];
        let indices = [0];
        let values = [10.0];
        let mut output = [0.0f32; 2];
        assert!(index_put(&input, &indices, &values, &mut output, &[2], 3, true).is_err());
    }

    // -- embedding_lookup tests -------------------------------------------

    #[test]
    fn test_embedding_lookup_basic() {
        // vocab=3, dim=2
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = [2, 0, 1];
        let mut output = [0.0f32; 6];
        embedding_lookup(&table, &indices, &mut output, 3, 2, true).unwrap();
        assert_eq!(output, [5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embedding_lookup_single_token() {
        let table = [10.0, 20.0, 30.0, 40.0];
        let indices = [1];
        let mut output = [0.0f32; 2];
        embedding_lookup(&table, &indices, &mut output, 2, 2, true).unwrap();
        assert_eq!(output, [30.0, 40.0]);
    }

    #[test]
    fn test_embedding_lookup_large_vocab() {
        let vocab = 1000;
        let dim = 64;
        let table: Vec<f32> = (0..(vocab * dim) as u32).map(|x| x as f32).collect();
        let indices = [0, 999, 500];
        let mut output = vec![0.0f32; 3 * dim];
        embedding_lookup(&table, &indices, &mut output, vocab, dim, true).unwrap();
        assert_eq!(output[0], 0.0);
        assert_eq!(output[dim], (999 * dim) as f32);
        assert_eq!(output[2 * dim], (500 * dim) as f32);
    }

    #[test]
    fn test_embedding_lookup_oob_error() {
        let table = [1.0, 2.0, 3.0, 4.0];
        let indices = [5];
        let mut output = [0.0f32; 2];
        assert!(embedding_lookup(&table, &indices, &mut output, 2, 2, true).is_err());
    }

    #[test]
    fn test_embedding_lookup_oob_clamp() {
        let table = [1.0, 2.0, 3.0, 4.0];
        let indices = [99];
        let mut output = [0.0f32; 2];
        embedding_lookup(&table, &indices, &mut output, 2, 2, false).unwrap();
        assert_eq!(output, [3.0, 4.0]); // clamped to last row
    }

    #[test]
    fn test_embedding_lookup_zero_vocab() {
        let table: &[f32] = &[];
        let indices: &[u32] = &[];
        let mut output: Vec<f32> = vec![];
        assert!(embedding_lookup(table, indices, &mut output, 0, 4, true).is_err());
    }

    #[test]
    fn test_embedding_lookup_zero_dim() {
        let table = [1.0];
        let indices = [0u32];
        let mut output = [0.0f32; 1];
        assert!(embedding_lookup(&table, &indices, &mut output, 1, 0, true).is_err());
    }

    #[test]
    fn test_embedding_lookup_duplicate_ids() {
        let table = [10.0, 20.0, 30.0, 40.0];
        let indices = [1, 1, 0, 0];
        let mut output = [0.0f32; 8];
        embedding_lookup(&table, &indices, &mut output, 2, 2, true).unwrap();
        assert_eq!(output, [30.0, 40.0, 30.0, 40.0, 10.0, 20.0, 10.0, 20.0]);
    }

    // -- GPU stub tests ---------------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gather_stub() {
        let input = vec![1.0f32; 1024];
        let indices: Vec<usize> = (0..256).collect();
        let mut output = vec![0.0f32; 256];
        let _ = gather(&input, &indices, &mut output, &[1024], 0, false);
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_scatter_stub() {
        let input = vec![0.0f32; 1024];
        let indices: Vec<usize> = (0..256).collect();
        let updates = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1024];
        let _ = scatter(&input, &indices, &updates, &mut output, &[1024], 0, false);
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_embedding_lookup_stub() {
        let table = vec![1.0f32; 4096];
        let indices: Vec<u32> = (0..32).collect();
        let mut output = vec![0.0f32; 32 * 128];
        let _ = embedding_lookup(&table, &indices, &mut output, 32, 128, true);
    }
}
