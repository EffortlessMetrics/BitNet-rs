//! Parallel reduction CUDA kernels for sum, max, min, mean, and L2 norm.
//!
//! # Kernel strategy
//!
//! Reduction operations are fundamental building blocks for softmax (max + sum),
//! attention scoring (sum), and LayerNorm (mean + L2 norm). This module provides
//! flat, row-wise, and column-wise reductions over f32 data.
//!
//! ## GPU path (feature `gpu` or `cuda`)
//!
//! The CUDA kernel uses a tree-based shared-memory reduction:
//!
//! 1. Each thread computes a partial reduction over its assigned elements.
//! 2. Partial results are stored in shared memory.
//! 3. A tree reduction in shared memory produces the final result per block.
//! 4. For row-wise reductions, one block handles one row. For column-wise
//!    reductions, one block handles one column.
//!
//! Target: full warp utilisation for typical BitNet dimensions (2048–4096).
//!
//! ## CPU fallback
//!
//! When CUDA is unavailable, the CPU path provides scalar implementations of
//! all reduction operations. These serve as reference for testing and for
//! environments without a GPU.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Reduction operation enum
// ---------------------------------------------------------------------------

/// Specifies which reduction operation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum of all elements.
    Sum,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
    /// Arithmetic mean (sum / count).
    Mean,
    /// L2 norm: sqrt(sum of squares).
    L2Norm,
}

impl ReductionOp {
    /// Identity element for this reduction (used to initialise accumulators).
    pub(crate) fn identity(self) -> f32 {
        match self {
            Self::Sum | Self::Mean | Self::L2Norm => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
        }
    }

    /// Combine two partial results.
    pub(crate) fn combine(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Sum | Self::Mean => a + b,
            Self::Max => a.max(b),
            Self::Min => a.min(b),
            Self::L2Norm => a + b, // accumulate squares, sqrt at finalise
        }
    }

    /// Pre-process an element before accumulation.
    pub(crate) fn map_element(self, x: f32) -> f32 {
        match self {
            Self::L2Norm => x * x,
            _ => x,
        }
    }

    /// Post-process the accumulated result.
    pub(crate) fn finalise(self, acc: f32, count: usize) -> f32 {
        match self {
            Self::Mean => {
                if count == 0 {
                    0.0
                } else {
                    acc / count as f32
                }
            }
            Self::L2Norm => acc.sqrt(),
            _ => acc,
        }
    }
}

// ---------------------------------------------------------------------------
// PTX source for the CUDA reduction kernel
// ---------------------------------------------------------------------------

/// Inline PTX-compatible CUDA C source for parallel reductions.
///
/// The kernel processes one row per thread-block. The `op` parameter selects
/// the reduction type: 0=Sum, 1=Max, 2=Min, 3=Mean, 4=L2Norm.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const REDUCTION_KERNEL_SRC: &str = r#"
extern "C" __global__ void reduce_rows_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int cols,
    int op)
{
    int row = blockIdx.x;
    const float* x = input + row * cols;

    extern __shared__ float sdata[];

    // Identity values per op
    float identity;
    switch (op) {
        case 1: identity = -3.402823466e+38f; break; // Max: -FLT_MAX
        case 2: identity =  3.402823466e+38f; break; // Min: FLT_MAX
        default: identity = 0.0f; break;              // Sum/Mean/L2Norm
    }

    float local_acc = identity;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        // Pre-process: square for L2Norm
        if (op == 4) val = val * val;

        switch (op) {
            case 1: local_acc = fmaxf(local_acc, val); break;
            case 2: local_acc = fminf(local_acc, val); break;
            default: local_acc += val; break;
        }
    }

    sdata[threadIdx.x] = local_acc;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            switch (op) {
                case 1:
                    sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x],
                                                sdata[threadIdx.x + stride]);
                    break;
                case 2:
                    sdata[threadIdx.x] = fminf(sdata[threadIdx.x],
                                                sdata[threadIdx.x + stride]);
                    break;
                default:
                    sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                    break;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float result = sdata[0];
        // Post-process
        if (op == 3) result = result / (float)cols; // Mean
        if (op == 4) result = sqrtf(result);         // L2Norm
        output[row] = result;
    }
}
"#;

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for reduction kernels.
#[derive(Debug, Clone)]
pub struct ReductionConfig {
    /// Number of elements to reduce (or columns for row-wise).
    pub reduce_dim: usize,
    /// Number of independent reductions (rows for row-wise, cols for col-wise).
    pub n_reductions: usize,
    /// Threads per block — typically `min(reduce_dim, 1024)`.
    pub threads_per_block: u32,
    /// The reduction operation to perform.
    pub op: ReductionOp,
}

impl ReductionConfig {
    /// Create a configuration for reducing `reduce_dim` elements across
    /// `n_reductions` independent groups.
    pub fn new(reduce_dim: usize, n_reductions: usize, op: ReductionOp) -> Result<Self> {
        if reduce_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "reduction dimension must be non-zero".to_string(),
            }
            .into());
        }
        if n_reductions == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "number of reductions must be non-zero".to_string(),
            }
            .into());
        }

        let threads_per_block = (reduce_dim as u32).min(1024);

        Ok(Self { reduce_dim, n_reductions, threads_per_block, op })
    }

    /// CUDA grid dimensions `(n_reductions, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.n_reductions as u32, 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }

    /// Bytes of shared memory required for the reduction tree.
    pub fn shared_mem_bytes(&self) -> u32 {
        self.threads_per_block * 4
    }
}

// ---------------------------------------------------------------------------
// CPU implementations
// ---------------------------------------------------------------------------

/// Reduce a flat f32 slice to a single scalar using the given operation.
///
/// Returns the identity element for empty slices (0 for Sum/Mean/L2Norm,
/// -inf for Max, inf for Min).
pub fn reduce_f32(data: &[f32], op: ReductionOp) -> f32 {
    if data.is_empty() {
        return op.identity();
    }

    let acc = data.iter().map(|&x| op.map_element(x)).fold(op.identity(), |a, b| op.combine(a, b));

    op.finalise(acc, data.len())
}

/// Row-wise reduction of a `[rows, cols]` matrix.
///
/// Returns a vector of length `rows`, where each element is the reduction
/// of the corresponding row.
///
/// # Errors
///
/// Returns `KernelError::InvalidArguments` if `matrix.len() != rows * cols`
/// or if dimensions are zero.
pub fn reduce_rows_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    op: ReductionOp,
) -> Result<Vec<f32>> {
    validate_matrix(matrix, rows, cols)?;

    let mut result = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let row_data = &matrix[start..start + cols];
        result.push(reduce_f32(row_data, op));
    }
    Ok(result)
}

/// Column-wise reduction of a `[rows, cols]` matrix.
///
/// Returns a vector of length `cols`, where each element is the reduction
/// of the corresponding column.
///
/// # Errors
///
/// Returns `KernelError::InvalidArguments` if `matrix.len() != rows * cols`
/// or if dimensions are zero.
pub fn reduce_cols_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    op: ReductionOp,
) -> Result<Vec<f32>> {
    validate_matrix(matrix, rows, cols)?;

    let mut result = vec![op.identity(); cols];
    for row in 0..rows {
        let start = row * cols;
        for col in 0..cols {
            let val = op.map_element(matrix[start + col]);
            result[col] = op.combine(result[col], val);
        }
    }

    // Finalise each column
    for v in &mut result {
        *v = op.finalise(*v, rows);
    }
    Ok(result)
}

/// Validate that `matrix` has the expected length for `[rows, cols]`.
fn validate_matrix(matrix: &[f32], rows: usize, cols: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("matrix dimensions must be non-zero: rows={rows}, cols={cols}"),
        }
        .into());
    }
    let expected = rows * cols;
    if matrix.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "matrix length mismatch: expected {} ({}×{}), got {}",
                expected,
                rows,
                cols,
                matrix.len(),
            ),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA dispatch (feature-gated)
// ---------------------------------------------------------------------------

/// Dispatch row-wise reduction to the CUDA device.
///
/// Compiles the PTX kernel at first invocation, transfers data to the device,
/// launches one block per row, and copies the result back.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_reduce_rows_cuda(
    input: &[f32],
    output: &mut [f32],
    config: &ReductionConfig,
) -> Result<()> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    let expected_in = config.n_reductions * config.reduce_dim;
    if input.len() != expected_in {
        return Err(KernelError::InvalidArguments {
            reason: format!("input length mismatch: expected {expected_in}, got {}", input.len(),),
        }
        .into());
    }
    if output.len() != config.n_reductions {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length mismatch: expected {}, got {}",
                config.n_reductions,
                output.len(),
            ),
        }
        .into());
    }

    log::debug!(
        "Reduction CUDA dispatch: reduce_dim={}, n_reductions={}, op={:?}",
        config.reduce_dim,
        config.n_reductions,
        config.op,
    );

    let ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
        reason: format!("failed to acquire CUDA device 0: {e:?}"),
    })?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(REDUCTION_KERNEL_SRC).map_err(|e| KernelError::GpuError {
        reason: format!("NVRTC compilation failed: {e:?}"),
    })?;

    let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
        reason: format!("failed to load PTX module: {e:?}"),
    })?;

    let func = module.load_function("reduce_rows_f32").map_err(|e| KernelError::GpuError {
        reason: format!("reduce_rows_f32 function not found: {e:?}"),
    })?;

    let input_dev = stream.memcpy_stod(input).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy input to device: {e:?}"),
    })?;

    let mut output_dev: CudaSlice<f32> = stream.alloc_zeros(config.n_reductions).map_err(|e| {
        KernelError::GpuError { reason: format!("failed to allocate output on device: {e:?}") }
    })?;

    let (gx, gy, gz) = config.grid_dim();
    let (bx, by, bz) = config.block_dim();
    let launch_cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (bx, by, bz),
        shared_mem_bytes: config.shared_mem_bytes(),
    };

    let cols_arg = config.reduce_dim as i32;
    let op_arg: i32 = match config.op {
        ReductionOp::Sum => 0,
        ReductionOp::Max => 1,
        ReductionOp::Min => 2,
        ReductionOp::Mean => 3,
        ReductionOp::L2Norm => 4,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_dev);
    builder.arg(&mut output_dev);
    builder.arg(&cols_arg);
    builder.arg(&op_arg);

    // Safety: kernel signature matches the CUDA source; buffers validated.
    unsafe { builder.launch(launch_cfg) }.map_err(|e| KernelError::GpuError {
        reason: format!("CUDA kernel launch failed: {e:?}"),
    })?;

    stream.synchronize().map_err(|e| KernelError::GpuError {
        reason: format!("stream synchronize failed: {e:?}"),
    })?;

    let output_host: Vec<f32> = stream.memcpy_dtov(&output_dev).map_err(|e| {
        KernelError::GpuError { reason: format!("failed to copy output from device: {e:?}") }
    })?;

    output.copy_from_slice(&output_host);

    Ok(())
}

// ---------------------------------------------------------------------------
// Unified dispatch entry points
// ---------------------------------------------------------------------------

/// Reduce a flat f32 slice with automatic CPU/GPU dispatch.
///
/// On GPU builds with CUDA available, dispatches as a single-row reduction.
/// Otherwise uses the CPU scalar path.
pub fn launch_reduce_f32(data: &[f32], op: ReductionOp) -> Result<f32> {
    if data.is_empty() {
        return Ok(op.identity());
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let config = ReductionConfig::new(data.len(), 1, op)?;
        let mut output = vec![0.0f32; 1];
        match launch_reduce_rows_cuda(data, &mut output, &config) {
            Ok(()) => {
                log::debug!("Reduction completed on CUDA (n={})", data.len());
                return Ok(output[0]);
            }
            Err(e) => {
                log::warn!("CUDA reduction failed, falling back to CPU: {e}");
            }
        }
    }

    Ok(reduce_f32(data, op))
}

/// Row-wise reduction with automatic CPU/GPU dispatch.
///
/// Returns a vector of length `rows`.
pub fn launch_reduce_rows_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    op: ReductionOp,
) -> Result<Vec<f32>> {
    validate_matrix(matrix, rows, cols)?;

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let config = ReductionConfig::new(cols, rows, op)?;
        let mut output = vec![0.0f32; rows];
        match launch_reduce_rows_cuda(matrix, &mut output, &config) {
            Ok(()) => {
                log::debug!("Row reduction completed on CUDA ({}×{})", rows, cols,);
                return Ok(output);
            }
            Err(e) => {
                log::warn!("CUDA row reduction failed, falling back to CPU: {e}");
            }
        }
    }

    reduce_rows_f32(matrix, rows, cols, op)
}

/// Column-wise reduction with automatic CPU/GPU dispatch.
///
/// Returns a vector of length `cols`.
///
/// Note: the column-wise kernel requires transposition on GPU, so the CPU
/// fallback is used unless a dedicated column-reduction kernel is provided.
pub fn launch_reduce_cols_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    op: ReductionOp,
) -> Result<Vec<f32>> {
    // Column reduction uses the CPU path; a dedicated CUDA kernel for
    // column-wise reduction can be added in a future iteration.
    reduce_cols_f32(matrix, rows, cols, op)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ReductionOp unit tests -------------------------------------------

    #[test]
    fn test_reduction_op_identity_values() {
        assert_eq!(ReductionOp::Sum.identity(), 0.0);
        assert_eq!(ReductionOp::Mean.identity(), 0.0);
        assert_eq!(ReductionOp::L2Norm.identity(), 0.0);
        assert_eq!(ReductionOp::Max.identity(), f32::NEG_INFINITY);
        assert_eq!(ReductionOp::Min.identity(), f32::INFINITY);
    }

    #[test]
    fn test_reduction_op_combine() {
        assert_eq!(ReductionOp::Sum.combine(3.0, 4.0), 7.0);
        assert_eq!(ReductionOp::Max.combine(3.0, 4.0), 4.0);
        assert_eq!(ReductionOp::Min.combine(3.0, 4.0), 3.0);
        assert_eq!(ReductionOp::Mean.combine(3.0, 4.0), 7.0);
        assert_eq!(ReductionOp::L2Norm.combine(9.0, 16.0), 25.0);
    }

    // -- Flat reduction tests ---------------------------------------------

    #[test]
    fn test_reduce_sum_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = reduce_f32(&data, ReductionOp::Sum);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max_basic() {
        let data = vec![1.0, 5.0, 3.0, 2.0];
        let result = reduce_f32(&data, ReductionOp::Max);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_min_basic() {
        let data = vec![4.0, 1.0, 3.0, 2.0];
        let result = reduce_f32(&data, ReductionOp::Min);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_mean_basic() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let result = reduce_f32(&data, ReductionOp::Mean);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_l2norm_basic() {
        // L2 norm of [3, 4] = sqrt(9 + 16) = 5
        let data = vec![3.0, 4.0];
        let result = reduce_f32(&data, ReductionOp::L2Norm);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_empty_slice() {
        assert_eq!(reduce_f32(&[], ReductionOp::Sum), 0.0);
        assert_eq!(reduce_f32(&[], ReductionOp::Max), f32::NEG_INFINITY);
        assert_eq!(reduce_f32(&[], ReductionOp::Min), f32::INFINITY);
        assert_eq!(reduce_f32(&[], ReductionOp::Mean), 0.0);
        assert_eq!(reduce_f32(&[], ReductionOp::L2Norm), 0.0);
    }

    #[test]
    fn test_reduce_single_element() {
        let data = vec![42.0];
        assert!((reduce_f32(&data, ReductionOp::Sum) - 42.0).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::Max) - 42.0).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::Min) - 42.0).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::Mean) - 42.0).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::L2Norm) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_negative_values() {
        let data = vec![-3.0, -1.0, -4.0, -1.5];
        assert!((reduce_f32(&data, ReductionOp::Sum) - (-9.5)).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::Max) - (-1.0)).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::Min) - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_mixed_sign_values() {
        let data = vec![-2.0, 3.0, -1.0, 4.0];
        assert!((reduce_f32(&data, ReductionOp::Sum) - 4.0).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::Max) - 4.0).abs() < 1e-6);
        assert!((reduce_f32(&data, ReductionOp::Min) - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_large_array() {
        let n = 10_000;
        let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let expected_sum = (n * (n + 1)) as f32 / 2.0;
        let result = reduce_f32(&data, ReductionOp::Sum);
        assert!(
            (result - expected_sum).abs() / expected_sum < 1e-4,
            "expected ~{expected_sum}, got {result}"
        );

        let max = reduce_f32(&data, ReductionOp::Max);
        assert!((max - n as f32).abs() < 1e-6);

        let min = reduce_f32(&data, ReductionOp::Min);
        assert!((min - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_l2norm_unit_vector() {
        // Unit vector: L2 norm should be 1.0
        let n = 100;
        let val = 1.0 / (n as f32).sqrt();
        let data = vec![val; n];
        let result = reduce_f32(&data, ReductionOp::L2Norm);
        assert!((result - 1.0).abs() < 1e-5, "expected ~1.0, got {result}");
    }

    #[test]
    fn test_reduce_numerical_accuracy() {
        // Kahan-style: lots of small values that accumulate rounding error
        let n = 100_000;
        let data = vec![1e-4_f32; n];
        let expected = n as f32 * 1e-4;
        let result = reduce_f32(&data, ReductionOp::Sum);
        assert!(
            (result - expected).abs() / expected < 1e-3,
            "relative error too large: expected {expected}, got {result}"
        );
    }

    // -- Row-wise reduction tests -----------------------------------------

    #[test]
    fn test_reduce_rows_sum() {
        let matrix = vec![
            1.0, 2.0, 3.0, // row 0: sum = 6
            4.0, 5.0, 6.0, // row 1: sum = 15
        ];
        let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Sum).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_rows_max() {
        let matrix = vec![
            1.0, 5.0, 3.0, // row 0: max = 5
            4.0, 2.0, 6.0, // row 1: max = 6
        ];
        let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Max).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_rows_min() {
        let matrix = vec![
            3.0, 1.0, 2.0, // row 0: min = 1
            6.0, 4.0, 5.0, // row 1: min = 4
        ];
        let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Min).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_rows_mean() {
        let matrix = vec![
            2.0, 4.0, 6.0, // row 0: mean = 4
            1.0, 3.0, 5.0, // row 1: mean = 3
        ];
        let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Mean).unwrap();
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_rows_l2norm() {
        let matrix = vec![
            3.0, 4.0, // row 0: sqrt(9+16) = 5
            5.0, 12.0, // row 1: sqrt(25+144) = 13
        ];
        let result = reduce_rows_f32(&matrix, 2, 2, ReductionOp::L2Norm).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-5);
        assert!((result[1] - 13.0).abs() < 1e-5);
    }

    // -- Column-wise reduction tests --------------------------------------

    #[test]
    fn test_reduce_cols_sum() {
        let matrix = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        // col sums: [5, 7, 9]
        let result = reduce_cols_f32(&matrix, 2, 3, ReductionOp::Sum).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 7.0).abs() < 1e-6);
        assert!((result[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_cols_max() {
        let matrix = vec![
            1.0, 5.0, 3.0, // row 0
            4.0, 2.0, 6.0, // row 1
        ];
        // col maxes: [4, 5, 6]
        let result = reduce_cols_f32(&matrix, 2, 3, ReductionOp::Max).unwrap();
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_cols_mean() {
        let matrix = vec![
            2.0, 4.0, // row 0
            6.0, 8.0, // row 1
        ];
        // col means: [4, 6]
        let result = reduce_cols_f32(&matrix, 2, 2, ReductionOp::Mean).unwrap();
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_cols_l2norm() {
        let matrix = vec![
            3.0, 5.0, // row 0
            4.0, 12.0, // row 1
        ];
        // col L2: [sqrt(9+16)=5, sqrt(25+144)=13]
        let result = reduce_cols_f32(&matrix, 2, 2, ReductionOp::L2Norm).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-5);
        assert!((result[1] - 13.0).abs() < 1e-5);
    }

    // -- Validation / error tests -----------------------------------------

    #[test]
    fn test_reduce_rows_dimension_mismatch() {
        let matrix = vec![1.0, 2.0, 3.0]; // 3 elements
        let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Sum); // expects 6
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_cols_dimension_mismatch() {
        let matrix = vec![1.0, 2.0]; // 2 elements
        let result = reduce_cols_f32(&matrix, 2, 2, ReductionOp::Sum); // expects 4
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_rows_zero_rows() {
        let matrix = vec![1.0, 2.0];
        assert!(reduce_rows_f32(&matrix, 0, 2, ReductionOp::Sum).is_err());
    }

    #[test]
    fn test_reduce_rows_zero_cols() {
        let matrix = vec![1.0, 2.0];
        assert!(reduce_rows_f32(&matrix, 2, 0, ReductionOp::Sum).is_err());
    }

    #[test]
    fn test_config_rejects_zero_reduce_dim() {
        assert!(ReductionConfig::new(0, 1, ReductionOp::Sum).is_err());
    }

    #[test]
    fn test_config_rejects_zero_n_reductions() {
        assert!(ReductionConfig::new(4, 0, ReductionOp::Sum).is_err());
    }

    // -- Config tests -----------------------------------------------------

    #[test]
    fn test_config_grid_and_block_dim() {
        let cfg = ReductionConfig::new(2048, 4, ReductionOp::Sum).unwrap();
        assert_eq!(cfg.grid_dim(), (4, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1)); // capped
        assert_eq!(cfg.shared_mem_bytes(), 1024 * 4);
    }

    #[test]
    fn test_config_small_reduce_dim() {
        let cfg = ReductionConfig::new(64, 8, ReductionOp::Max).unwrap();
        assert_eq!(cfg.block_dim(), (64, 1, 1));
        assert_eq!(cfg.shared_mem_bytes(), 64 * 4);
    }

    // -- Dispatch tests ---------------------------------------------------

    #[test]
    fn test_launch_reduce_f32_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = launch_reduce_f32(&data, ReductionOp::Sum).unwrap();
        assert!((result - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_launch_reduce_f32_empty() {
        let result = launch_reduce_f32(&[], ReductionOp::Sum).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_launch_reduce_rows_matches_cpu() {
        let matrix = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ];
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
            ReductionOp::L2Norm,
        ] {
            let dispatch = launch_reduce_rows_f32(&matrix, 3, 4, op).unwrap();
            let cpu = reduce_rows_f32(&matrix, 3, 4, op).unwrap();
            for (i, (&d, &c)) in dispatch.iter().zip(cpu.iter()).enumerate() {
                assert!((d - c).abs() < 1e-5, "op={op:?}, row {i}: dispatch={d}, cpu={c}");
            }
        }
    }

    #[test]
    fn test_launch_reduce_cols_matches_cpu() {
        let matrix = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
            ReductionOp::L2Norm,
        ] {
            let dispatch = launch_reduce_cols_f32(&matrix, 2, 3, op).unwrap();
            let cpu = reduce_cols_f32(&matrix, 2, 3, op).unwrap();
            for (i, (&d, &c)) in dispatch.iter().zip(cpu.iter()).enumerate() {
                assert!((d - c).abs() < 1e-5, "op={op:?}, col {i}: dispatch={d}, cpu={c}");
            }
        }
    }

    // -- GPU-only tests ---------------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_reduction_sum() {
        let data: Vec<f32> = (1..=1024).map(|i| i as f32).collect();
        let result = launch_reduce_f32(&data, ReductionOp::Sum).unwrap();
        let expected = 1024.0 * 1025.0 / 2.0;
        assert!((result - expected).abs() / expected < 1e-4, "expected {expected}, got {result}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_reduction_matches_cpu() {
        let n = 4096;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 20.0).collect();
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
            ReductionOp::L2Norm,
        ] {
            let gpu = launch_reduce_f32(&data, op).unwrap();
            let cpu = reduce_f32(&data, op);
            assert!((gpu - cpu).abs() < 1e-2, "op={op:?}: gpu={gpu}, cpu={cpu}");
        }
    }
}
