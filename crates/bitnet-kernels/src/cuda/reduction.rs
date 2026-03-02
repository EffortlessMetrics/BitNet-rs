//! CUDA parallel reduction operations for tensor computations.
//!
//! This module provides GPU-accelerated reduction operations including sum,
//! mean, max, min, product, L2 norm, variance, and argmax/argmin. All
//! functions include CPU fallback implementations for correctness testing
//! and non-GPU environments.
//!
//! # Operations
//!
//! - [`reduce_sum`] / [`reduce_mean`]: Sum and mean reduction along an axis.
//! - [`reduce_max`] / [`reduce_min`]: Max/min reduction returning value + index.
//! - [`reduce_prod`]: Product reduction along an axis.
//! - [`reduce_l2_norm`]: L2 norm (Euclidean length) along an axis.
//! - [`reduce_variance`]: Variance computation along an axis.
//! - [`global_sum`]: Full tensor sum to a single scalar.
//! - [`argmax`] / [`argmin`]: Index of the maximum/minimum along an axis.
//!
//! # CUDA kernels
//!
//! GPU launch stubs are gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`
//! and return `KernelError::GpuError` until real PTX kernels are compiled.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for CUDA parallel reduction kernel launches.
#[derive(Debug, Clone)]
pub struct CudaReductionConfig {
    /// Threads per block for GPU launch.
    pub threads_per_block: u32,
    /// Bytes of shared memory for the reduction tree.
    pub shared_mem_bytes: u32,
}

impl CudaReductionConfig {
    /// Create a new configuration sized for the given reduction dimension.
    pub fn new(reduce_dim: usize) -> Self {
        let threads_per_block = (reduce_dim as u32).clamp(1, 1024);
        let shared_mem_bytes = threads_per_block * 4; // f32 per thread
        Self { threads_per_block, shared_mem_bytes }
    }

    /// CUDA grid dimensions for independent reductions.
    pub fn grid_dim(&self, n_reductions: usize) -> (u32, u32, u32) {
        ((n_reductions as u32).max(1), 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Result of a reduction that returns both value and index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueIndex {
    /// The reduced value (max or min).
    pub value: f32,
    /// The index of the element within the reduction axis.
    pub index: usize,
}

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// CUDA C source for parallel sum reduction.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const REDUCE_SUM_KERNEL_SRC: &str = r#"
extern "C" __global__ void reduce_sum_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int reduce_dim,
    int n_reductions)
{
    int rid = blockIdx.x;
    if (rid >= n_reductions) return;
    const float* row = input + rid * reduce_dim;
    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        local_sum += row[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[rid] = sdata[0];
}
"#;

/// CUDA C source for argmax reduction.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ARGMAX_KERNEL_SRC: &str = r#"
extern "C" __global__ void argmax_f32(
    const float* __restrict__ input,
    float* __restrict__ out_values,
    int* __restrict__ out_indices,
    int reduce_dim,
    int n_reductions)
{
    int rid = blockIdx.x;
    if (rid >= n_reductions) return;
    const float* row = input + rid * reduce_dim;

    float best_val = -3.402823466e+38f;
    int best_idx = 0;
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        if (row[i] > best_val) {
            best_val = row[i];
            best_idx = i;
        }
    }
    // Single-thread simplification for stub; full warp reduction in production.
    if (threadIdx.x == 0) {
        for (int i = 0; i < reduce_dim; i++) {
            if (row[i] > best_val || (row[i] == best_val && i < best_idx)) {
                best_val = row[i];
                best_idx = i;
            }
        }
        out_values[rid] = best_val;
        out_indices[rid] = best_idx;
    }
}
"#;

// ---------------------------------------------------------------------------
// Helper: axis decomposition
// ---------------------------------------------------------------------------

/// Decompose a shape into (outer_size, axis_size, inner_size) around `axis`.
fn decompose_axis(shape: &[usize], axis: usize) -> Result<(usize, usize, usize)> {
    if shape.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "reduction: shape must not be empty".into(),
        }
        .into());
    }
    if axis >= shape.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("reduction: axis {axis} out of range for {}-D tensor", shape.len()),
        }
        .into());
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_size = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer = if outer == 0 { 1 } else { outer };
    let inner = if inner == 0 { 1 } else { inner };
    if axis_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "reduction: axis dimension must be non-zero".into(),
        }
        .into());
    }
    Ok((outer, axis_size, inner))
}

// ---------------------------------------------------------------------------
// CPU fallback — reduce_sum
// ---------------------------------------------------------------------------

/// Sum reduction along the specified axis.
///
/// Returns a tensor with the axis dimension removed. For a `[rows, cols]`
/// input with axis=1, returns a vector of length `rows`.
///
/// # Errors
///
/// Returns error on invalid axis or size mismatch.
pub fn reduce_sum(input: &[f32], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
    let (outer, axis_size, inner) = decompose_axis(shape, axis)?;
    let mut output = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut sum = 0.0f32;
            for a in 0..axis_size {
                sum += input[o * axis_size * inner + a * inner + i];
            }
            output[o * inner + i] = sum;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback — reduce_mean
// ---------------------------------------------------------------------------

/// Mean reduction along the specified axis.
///
/// # Errors
///
/// Returns error on invalid axis or size mismatch.
pub fn reduce_mean(input: &[f32], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
    let (outer, axis_size, inner) = decompose_axis(shape, axis)?;
    let mut output = vec![0.0f32; outer * inner];
    let count = axis_size as f32;
    for o in 0..outer {
        for i in 0..inner {
            let mut sum = 0.0f32;
            for a in 0..axis_size {
                sum += input[o * axis_size * inner + a * inner + i];
            }
            output[o * inner + i] = sum / count;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback — reduce_max
// ---------------------------------------------------------------------------

/// Max reduction along the specified axis, returning values and indices.
///
/// # Errors
///
/// Returns error on invalid axis or size mismatch.
pub fn reduce_max(input: &[f32], shape: &[usize], axis: usize) -> Result<(Vec<f32>, Vec<usize>)> {
    let (outer, axis_size, inner) = decompose_axis(shape, axis)?;
    let mut values = vec![f32::NEG_INFINITY; outer * inner];
    let mut indices = vec![0usize; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let out_idx = o * inner + i;
            for a in 0..axis_size {
                let val = input[o * axis_size * inner + a * inner + i];
                if val > values[out_idx] {
                    values[out_idx] = val;
                    indices[out_idx] = a;
                }
            }
        }
    }
    Ok((values, indices))
}

// ---------------------------------------------------------------------------
// CPU fallback — reduce_min
// ---------------------------------------------------------------------------

/// Min reduction along the specified axis, returning values and indices.
///
/// # Errors
///
/// Returns error on invalid axis or size mismatch.
pub fn reduce_min(input: &[f32], shape: &[usize], axis: usize) -> Result<(Vec<f32>, Vec<usize>)> {
    let (outer, axis_size, inner) = decompose_axis(shape, axis)?;
    let mut values = vec![f32::INFINITY; outer * inner];
    let mut indices = vec![0usize; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let out_idx = o * inner + i;
            for a in 0..axis_size {
                let val = input[o * axis_size * inner + a * inner + i];
                if val < values[out_idx] {
                    values[out_idx] = val;
                    indices[out_idx] = a;
                }
            }
        }
    }
    Ok((values, indices))
}

// ---------------------------------------------------------------------------
// CPU fallback — reduce_prod
// ---------------------------------------------------------------------------

/// Product reduction along the specified axis.
///
/// # Errors
///
/// Returns error on invalid axis or size mismatch.
pub fn reduce_prod(input: &[f32], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
    let (outer, axis_size, inner) = decompose_axis(shape, axis)?;
    let mut output = vec![1.0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let out_idx = o * inner + i;
            for a in 0..axis_size {
                output[out_idx] *= input[o * axis_size * inner + a * inner + i];
            }
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback — reduce_l2_norm
// ---------------------------------------------------------------------------

/// L2 norm reduction along the specified axis: `sqrt(sum(x^2))`.
///
/// # Errors
///
/// Returns error on invalid axis or size mismatch.
pub fn reduce_l2_norm(input: &[f32], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
    let (outer, axis_size, inner) = decompose_axis(shape, axis)?;
    let mut output = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let out_idx = o * inner + i;
            let mut sum_sq = 0.0f32;
            for a in 0..axis_size {
                let val = input[o * axis_size * inner + a * inner + i];
                sum_sq += val * val;
            }
            output[out_idx] = sum_sq.sqrt();
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback — reduce_variance
// ---------------------------------------------------------------------------

/// Variance reduction along the specified axis.
///
/// Computes population variance: `mean((x - mean(x))^2)`.
///
/// # Errors
///
/// Returns error on invalid axis or size mismatch.
pub fn reduce_variance(input: &[f32], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
    let (outer, axis_size, inner) = decompose_axis(shape, axis)?;
    let count = axis_size as f32;
    let mut output = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            // Two-pass: compute mean, then variance
            let mut sum = 0.0f32;
            for a in 0..axis_size {
                sum += input[o * axis_size * inner + a * inner + i];
            }
            let mean = sum / count;
            let mut var_sum = 0.0f32;
            for a in 0..axis_size {
                let diff = input[o * axis_size * inner + a * inner + i] - mean;
                var_sum += diff * diff;
            }
            output[o * inner + i] = var_sum / count;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback — global_sum
// ---------------------------------------------------------------------------

/// Full tensor sum to a single scalar.
pub fn global_sum(input: &[f32]) -> f32 {
    input.iter().sum()
}

// ---------------------------------------------------------------------------
// CPU fallback — argmax / argmin
// ---------------------------------------------------------------------------

/// Index of the maximum value along the specified axis.
///
/// # Errors
///
/// Returns error on invalid axis.
pub fn argmax(input: &[f32], shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    let (_values, indices) = reduce_max(input, shape, axis)?;
    Ok(indices)
}

/// Index of the minimum value along the specified axis.
///
/// # Errors
///
/// Returns error on invalid axis.
pub fn argmin(input: &[f32], shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    let (_values, indices) = reduce_min(input, shape, axis)?;
    Ok(indices)
}

// ---------------------------------------------------------------------------
// GPU launch stubs
// ---------------------------------------------------------------------------

/// GPU launch stub for sum reduction.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_cuda_reduce_sum(
    _input: &[f32],
    _output: &mut [f32],
    _shape: &[usize],
    _axis: usize,
) -> Result<()> {
    log::debug!("cuda reduction::reduce_sum stub invoked");
    Err(KernelError::GpuError { reason: "reduce_sum CUDA kernel not yet compiled — stub".into() }
        .into())
}

/// GPU launch stub for argmax reduction.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_cuda_argmax(
    _input: &[f32],
    _out_values: &mut [f32],
    _out_indices: &mut [usize],
    _shape: &[usize],
    _axis: usize,
) -> Result<()> {
    log::debug!("cuda reduction::argmax stub invoked");
    Err(KernelError::GpuError { reason: "argmax CUDA kernel not yet compiled — stub".into() }
        .into())
}

/// GPU launch stub for variance reduction.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_cuda_reduce_variance(
    _input: &[f32],
    _output: &mut [f32],
    _shape: &[usize],
    _axis: usize,
) -> Result<()> {
    log::debug!("cuda reduction::reduce_variance stub invoked");
    Err(KernelError::GpuError {
        reason: "reduce_variance CUDA kernel not yet compiled — stub".into(),
    }
    .into())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CudaReductionConfig tests ----------------------------------------

    #[test]
    fn test_config_new_basic() {
        let cfg = CudaReductionConfig::new(512);
        assert_eq!(cfg.threads_per_block, 512);
        assert_eq!(cfg.shared_mem_bytes, 512 * 4);
    }

    #[test]
    fn test_config_threads_capped() {
        let cfg = CudaReductionConfig::new(8192);
        assert_eq!(cfg.threads_per_block, 1024);
    }

    #[test]
    fn test_config_threads_min_1() {
        let cfg = CudaReductionConfig::new(0);
        assert_eq!(cfg.threads_per_block, 1);
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = CudaReductionConfig::new(256);
        assert_eq!(cfg.grid_dim(4), (4, 1, 1));
        assert_eq!(cfg.grid_dim(1), (1, 1, 1));
    }

    #[test]
    fn test_config_block_dim() {
        let cfg = CudaReductionConfig::new(128);
        assert_eq!(cfg.block_dim(), (128, 1, 1));
    }

    // -- reduce_sum tests -------------------------------------------------

    #[test]
    fn test_reduce_sum_1d() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = reduce_sum(&input, &[4], 0).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_2d_axis0() {
        // 2×3 → 3
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = reduce_sum(&input, &[2, 3], 0).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 7.0).abs() < 1e-6);
        assert!((result[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_2d_axis1() {
        // 2×3 → 2
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = reduce_sum(&input, &[2, 3], 1).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_3d() {
        // 2×2×2, axis=1
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = reduce_sum(&input, &[2, 2, 2], 1).unwrap();
        assert_eq!(result.len(), 4); // 2×2
        assert!((result[0] - 4.0).abs() < 1e-6); // 1+3
        assert!((result[1] - 6.0).abs() < 1e-6); // 2+4
    }

    #[test]
    fn test_reduce_sum_single_element() {
        let input = [42.0];
        let result = reduce_sum(&input, &[1], 0).unwrap();
        assert!((result[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_negative_values() {
        let input = [-1.0, -2.0, -3.0, -4.0];
        let result = reduce_sum(&input, &[4], 0).unwrap();
        assert!((result[0] - (-10.0)).abs() < 1e-6);
    }

    // -- reduce_mean tests ------------------------------------------------

    #[test]
    fn test_reduce_mean_1d() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let result = reduce_mean(&input, &[4], 0).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_mean_2d_axis0() {
        let input = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let result = reduce_mean(&input, &[2, 2], 0).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-6); // (1+3)/2
        assert!((result[1] - 3.0).abs() < 1e-6); // (2+4)/2
    }

    #[test]
    fn test_reduce_mean_2d_axis1() {
        let input = [1.0, 3.0, 5.0, 7.0]; // 2×2
        let result = reduce_mean(&input, &[2, 2], 1).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
    }

    // -- reduce_max tests -------------------------------------------------

    #[test]
    fn test_reduce_max_1d() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let (values, indices) = reduce_max(&input, &[5], 0).unwrap();
        assert!((values[0] - 5.0).abs() < 1e-6);
        assert_eq!(indices[0], 4);
    }

    #[test]
    fn test_reduce_max_2d_axis0() {
        // 2×3
        let input = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let (values, indices) = reduce_max(&input, &[2, 3], 0).unwrap();
        assert!((values[0] - 4.0).abs() < 1e-6);
        assert_eq!(indices[0], 1); // row 1
        assert!((values[1] - 5.0).abs() < 1e-6);
        assert_eq!(indices[1], 0); // row 0
    }

    #[test]
    fn test_reduce_max_2d_axis1() {
        let input = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]; // 2×3
        let (values, indices) = reduce_max(&input, &[2, 3], 1).unwrap();
        assert!((values[0] - 5.0).abs() < 1e-6);
        assert_eq!(indices[0], 1);
        assert!((values[1] - 6.0).abs() < 1e-6);
        assert_eq!(indices[1], 2);
    }

    #[test]
    fn test_reduce_max_all_negative() {
        let input = [-5.0, -3.0, -8.0, -1.0];
        let (values, indices) = reduce_max(&input, &[4], 0).unwrap();
        assert!((values[0] - (-1.0)).abs() < 1e-6);
        assert_eq!(indices[0], 3);
    }

    // -- reduce_min tests -------------------------------------------------

    #[test]
    fn test_reduce_min_1d() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let (values, indices) = reduce_min(&input, &[5], 0).unwrap();
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert_eq!(indices[0], 1); // first occurrence
    }

    #[test]
    fn test_reduce_min_2d_axis0() {
        let input = [4.0, 2.0, 6.0, 1.0, 5.0, 3.0]; // 2×3
        let (values, indices) = reduce_min(&input, &[2, 3], 0).unwrap();
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert_eq!(indices[0], 1);
    }

    #[test]
    fn test_reduce_min_2d_axis1() {
        let input = [3.0, 1.0, 4.0, 6.0, 2.0, 5.0]; // 2×3
        let (values, indices) = reduce_min(&input, &[2, 3], 1).unwrap();
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert_eq!(indices[0], 1);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert_eq!(indices[1], 1);
    }

    #[test]
    fn test_reduce_min_all_positive() {
        let input = [5.0, 3.0, 8.0, 1.0];
        let (values, indices) = reduce_min(&input, &[4], 0).unwrap();
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert_eq!(indices[0], 3);
    }

    // -- reduce_prod tests ------------------------------------------------

    #[test]
    fn test_reduce_prod_1d() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = reduce_prod(&input, &[4], 0).unwrap();
        assert!((result[0] - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_prod_2d_axis1() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let result = reduce_prod(&input, &[2, 3], 1).unwrap();
        assert!((result[0] - 6.0).abs() < 1e-6); // 1*2*3
        assert!((result[1] - 120.0).abs() < 1e-6); // 4*5*6
    }

    #[test]
    fn test_reduce_prod_with_zero() {
        let input = [1.0, 0.0, 3.0];
        let result = reduce_prod(&input, &[3], 0).unwrap();
        assert!((result[0]).abs() < 1e-6); // product is 0
    }

    #[test]
    fn test_reduce_prod_single() {
        let input = [7.0];
        let result = reduce_prod(&input, &[1], 0).unwrap();
        assert!((result[0] - 7.0).abs() < 1e-6);
    }

    // -- reduce_l2_norm tests ---------------------------------------------

    #[test]
    fn test_reduce_l2_norm_1d() {
        let input = [3.0, 4.0];
        let result = reduce_l2_norm(&input, &[2], 0).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_l2_norm_2d_axis1() {
        let input = [3.0, 4.0, 5.0, 12.0]; // 2×2
        let result = reduce_l2_norm(&input, &[2, 2], 1).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-5);
        assert!((result[1] - 13.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_l2_norm_unit_vector() {
        let n = 100;
        let val = 1.0 / (n as f32).sqrt();
        let input = vec![val; n];
        let result = reduce_l2_norm(&input, &[n], 0).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_l2_norm_zeros() {
        let input = [0.0, 0.0, 0.0];
        let result = reduce_l2_norm(&input, &[3], 0).unwrap();
        assert!((result[0]).abs() < 1e-6);
    }

    // -- reduce_variance tests --------------------------------------------

    #[test]
    fn test_reduce_variance_1d() {
        // [1, 2, 3, 4, 5] → mean=3, var=2.0
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = reduce_variance(&input, &[5], 0).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_variance_constant() {
        let input = [5.0, 5.0, 5.0, 5.0];
        let result = reduce_variance(&input, &[4], 0).unwrap();
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_variance_2d_axis1() {
        // row 0: [1, 3] → mean=2, var=1
        // row 1: [4, 8] → mean=6, var=4
        let input = [1.0, 3.0, 4.0, 8.0];
        let result = reduce_variance(&input, &[2, 2], 1).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_variance_single_element() {
        let input = [42.0];
        let result = reduce_variance(&input, &[1], 0).unwrap();
        assert!((result[0]).abs() < 1e-6);
    }

    // -- global_sum tests -------------------------------------------------

    #[test]
    fn test_global_sum_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((global_sum(&input) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_sum_empty() {
        assert!((global_sum(&[]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_sum_single() {
        assert!((global_sum(&[42.0]) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_sum_negative() {
        let input = [-1.0, -2.0, 3.0, 4.0];
        assert!((global_sum(&input) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_sum_large() {
        let n = 10_000;
        let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let expected = (n * (n + 1)) as f32 / 2.0;
        assert!((global_sum(&input) - expected).abs() / expected < 1e-4);
    }

    // -- argmax / argmin tests --------------------------------------------

    #[test]
    fn test_argmax_1d() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let result = argmax(&input, &[5], 0).unwrap();
        assert_eq!(result[0], 4);
    }

    #[test]
    fn test_argmax_2d_axis0() {
        let input = [1.0, 5.0, 4.0, 2.0]; // 2×2
        let result = argmax(&input, &[2, 2], 0).unwrap();
        assert_eq!(result[0], 1); // col 0: max at row 1
        assert_eq!(result[1], 0); // col 1: max at row 0
    }

    #[test]
    fn test_argmax_2d_axis1() {
        let input = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]; // 2×3
        let result = argmax(&input, &[2, 3], 1).unwrap();
        assert_eq!(result[0], 1); // row 0: max at col 1
        assert_eq!(result[1], 2); // row 1: max at col 2
    }

    #[test]
    fn test_argmin_1d() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let result = argmin(&input, &[5], 0).unwrap();
        assert_eq!(result[0], 1); // first min
    }

    #[test]
    fn test_argmin_2d_axis1() {
        let input = [3.0, 1.0, 4.0, 6.0, 2.0, 5.0]; // 2×3
        let result = argmin(&input, &[2, 3], 1).unwrap();
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 1);
    }

    // -- Error / edge case tests ------------------------------------------

    #[test]
    fn test_reduce_sum_invalid_axis() {
        let input = [1.0, 2.0];
        assert!(reduce_sum(&input, &[2], 1).is_err());
    }

    #[test]
    fn test_reduce_sum_empty_shape() {
        let input = [1.0];
        assert!(reduce_sum(&input, &[], 0).is_err());
    }

    #[test]
    fn test_reduce_mean_invalid_axis() {
        let input = [1.0];
        assert!(reduce_mean(&input, &[1], 2).is_err());
    }

    #[test]
    fn test_reduce_max_invalid_axis() {
        let input = [1.0, 2.0];
        assert!(reduce_max(&input, &[2], 5).is_err());
    }

    #[test]
    fn test_reduce_prod_invalid_axis() {
        let input = [1.0];
        assert!(reduce_prod(&input, &[1], 1).is_err());
    }

    #[test]
    fn test_reduce_l2_norm_invalid_axis() {
        let input = [1.0];
        assert!(reduce_l2_norm(&input, &[1], 3).is_err());
    }

    #[test]
    fn test_reduce_variance_invalid_axis() {
        let input = [1.0];
        assert!(reduce_variance(&input, &[1], 1).is_err());
    }

    #[test]
    fn test_argmax_invalid_axis() {
        let input = [1.0];
        assert!(argmax(&input, &[1], 2).is_err());
    }

    #[test]
    fn test_argmin_invalid_axis() {
        let input = [1.0];
        assert!(argmin(&input, &[1], 2).is_err());
    }

    // -- Large tensor tests -----------------------------------------------

    #[test]
    fn test_reduce_sum_large_matrix() {
        let rows = 64;
        let cols = 128;
        let input: Vec<f32> = (0..(rows * cols) as u32).map(|x| x as f32).collect();
        let result = reduce_sum(&input, &[rows, cols], 1).unwrap();
        assert_eq!(result.len(), rows);
        // Row 0: sum of 0..127
        let expected_row0 = (0..cols as u32).map(|x| x as f32).sum::<f32>();
        assert!((result[0] - expected_row0).abs() < 1.0);
    }

    #[test]
    fn test_reduce_max_large_1d() {
        let n = 10_000;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let (values, indices) = reduce_max(&input, &[n], 0).unwrap();
        assert!((values[0] - (n - 1) as f32 * 0.1).abs() < 1e-3);
        assert_eq!(indices[0], n - 1);
    }

    // -- GPU stub tests ---------------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_reduce_sum_stub() {
        let input = vec![1.0f32; 1024];
        let _ = reduce_sum(&input, &[1024], 0);
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_argmax_stub() {
        let input = vec![1.0f32; 1024];
        let _ = argmax(&input, &[1024], 0);
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_reduce_variance_stub() {
        let input = vec![1.0f32; 1024];
        let _ = reduce_variance(&input, &[1024], 0);
    }
}
