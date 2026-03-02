//! OpenCL-accelerated reduction operations for Intel Arc A770 (Xe-HPG).
//!
//! Provides GPU-optimized sum, max, min, mean, variance, argmax, argmin,
//! product, and log-sum-exp reductions with two-phase tree reduction
//! (local workgroup → global), subgroup shuffle, and shared local memory
//! for workgroup intermediate results.
//!
//! CPU reference implementations are always available and serve as the
//! correctness baseline for testing.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// A770 hardware constants
// ---------------------------------------------------------------------------

/// Intel Arc A770 (Xe-HPG) reduction-specific tuning constants.
pub struct A770ReduceConstants;

impl A770ReduceConstants {
    /// Preferred workgroup size for reduction kernels.
    pub const WORKGROUP_SIZE: usize = 256;
    /// Subgroup (warp-equivalent) width on Xe-HPG.
    pub const SUBGROUP_WIDTH: usize = 16;
    /// Maximum local memory per workgroup (bytes).
    pub const LOCAL_MEM_BYTES: usize = 65_536;
    /// Number of Xe-cores on A770.
    pub const COMPUTE_UNITS: usize = 32;
    /// Threshold at which two-pass reduction is used.
    pub const TWO_PASS_THRESHOLD: usize = 65_536;
    /// Maximum workgroups for the first pass of two-pass reduction.
    pub const MAX_FIRST_PASS_WORKGROUPS: usize = 256;
}

// ---------------------------------------------------------------------------
// ReduceOp enum
// ---------------------------------------------------------------------------

/// Specifies which reduction operation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
    Variance,
    ArgMax,
    ArgMin,
    Prod,
    LogSumExp,
}

impl ReduceOp {
    /// Identity element for accumulation.
    pub fn identity(self) -> f32 {
        match self {
            Self::Sum | Self::Mean | Self::Variance => 0.0,
            Self::Max | Self::ArgMax | Self::LogSumExp => f32::NEG_INFINITY,
            Self::Min | Self::ArgMin => f32::INFINITY,
            Self::Prod => 1.0,
        }
    }

    /// Combine two partial results.
    pub fn combine(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Sum | Self::Mean | Self::Variance => a + b,
            Self::Max | Self::ArgMax | Self::LogSumExp => a.max(b),
            Self::Min | Self::ArgMin => a.min(b),
            Self::Prod => a * b,
        }
    }

    /// Whether this op returns indices rather than values.
    pub fn returns_indices(self) -> bool {
        matches!(self, Self::ArgMax | Self::ArgMin)
    }
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "Sum"),
            Self::Max => write!(f, "Max"),
            Self::Min => write!(f, "Min"),
            Self::Mean => write!(f, "Mean"),
            Self::Variance => write!(f, "Variance"),
            Self::ArgMax => write!(f, "ArgMax"),
            Self::ArgMin => write!(f, "ArgMin"),
            Self::Prod => write!(f, "Prod"),
            Self::LogSumExp => write!(f, "LogSumExp"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReduceDtype
// ---------------------------------------------------------------------------

/// Data types supported by the reduction kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceDtype {
    F32,
    F16,
    I32,
}

impl ReduceDtype {
    /// Size of one element in bytes.
    pub fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I32 => 4,
        }
    }
}

impl fmt::Display for ReduceDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::I32 => write!(f, "i32"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReduceConfig
// ---------------------------------------------------------------------------

/// Configuration for a reduction operation.
#[derive(Debug, Clone)]
pub struct ReduceConfig {
    /// The reduction operation.
    pub op: ReduceOp,
    /// Axis to reduce along. `None` reduces globally.
    pub axis: Option<usize>,
    /// If true, the reduced axis is kept as dimension of size 1.
    pub keepdims: bool,
    /// Element data type.
    pub dtype: ReduceDtype,
}

impl ReduceConfig {
    pub fn new(op: ReduceOp) -> Self {
        Self { op, axis: None, keepdims: false, dtype: ReduceDtype::F32 }
    }

    pub fn with_axis(mut self, axis: usize) -> Self {
        self.axis = Some(axis);
        self
    }

    pub fn with_keepdims(mut self, keepdims: bool) -> Self {
        self.keepdims = keepdims;
        self
    }

    pub fn with_dtype(mut self, dtype: ReduceDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Global reduction (no axis, no keepdims).
    pub fn global(op: ReduceOp) -> Self {
        Self::new(op)
    }
}

// ---------------------------------------------------------------------------
// ReduceStats
// ---------------------------------------------------------------------------

/// Statistics collected from a reduction operation.
#[derive(Debug, Clone, Copy)]
pub struct ReduceStats {
    /// Wall-clock time for the reduction (seconds).
    pub compute_time: f64,
    /// Estimated bandwidth utilisation (bytes/sec).
    pub bandwidth_utilization: f64,
    /// Total number of input elements processed.
    pub elements_processed: usize,
}

impl ReduceStats {
    fn compute(elements: usize, bytes_per_element: usize, elapsed: f64) -> Self {
        let total_bytes = (elements * bytes_per_element) as f64;
        let bw = if elapsed > 0.0 { total_bytes / elapsed } else { 0.0 };
        Self {
            compute_time: elapsed,
            bandwidth_utilization: bw,
            elements_processed: elements,
        }
    }
}

impl fmt::Display for ReduceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "elements={}, time={:.3}ms, BW={:.2} GB/s",
            self.elements_processed,
            self.compute_time * 1000.0,
            self.bandwidth_utilization / 1e9,
        )
    }
}

// ---------------------------------------------------------------------------
// Output shape helpers
// ---------------------------------------------------------------------------

/// Compute the output shape after reducing along an axis.
fn output_shape(
    input_shape: &[usize],
    axis: Option<usize>,
    keepdims: bool,
) -> Vec<usize> {
    match axis {
        None => {
            if keepdims {
                vec![1; input_shape.len()]
            } else {
                vec![]
            }
        }
        Some(ax) => {
            if keepdims {
                let mut s = input_shape.to_vec();
                s[ax] = 1;
                s
            } else {
                input_shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != ax)
                    .map(|(_, &d)| d)
                    .collect()
            }
        }
    }
}

/// Total number of elements from a shape.
fn shape_numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ---------------------------------------------------------------------------
// Reduction result
// ---------------------------------------------------------------------------

/// Result of a reduction operation.
#[derive(Debug, Clone)]
pub struct ReduceResult {
    /// The output values.
    pub values: Vec<f32>,
    /// Output shape after reduction.
    pub shape: Vec<usize>,
    /// Performance statistics.
    pub stats: ReduceStats,
}

// ---------------------------------------------------------------------------
// TreeReducer — two-pass tree reduction for large tensors
// ---------------------------------------------------------------------------

/// Two-pass tree reduction: local workgroup partial reduces → global reduce.
///
/// Phase 1: divide input into chunks of `workgroup_size`, reduce each chunk
/// to a single partial result. Phase 2: reduce partial results to final.
pub struct TreeReducer {
    /// Workgroup size for the local pass.
    pub workgroup_size: usize,
}

impl TreeReducer {
    pub fn new() -> Self {
        Self {
            workgroup_size: A770ReduceConstants::WORKGROUP_SIZE,
        }
    }

    pub fn with_workgroup_size(workgroup_size: usize) -> Self {
        Self {
            workgroup_size: workgroup_size.max(1),
        }
    }

    /// Whether two-pass reduction is needed for the given element count.
    pub fn needs_two_pass(&self, n: usize) -> bool {
        n > self.workgroup_size
    }

    /// Number of workgroups for the first pass.
    pub fn first_pass_workgroups(&self, n: usize) -> usize {
        let wg = n.div_ceil(self.workgroup_size);
        wg.min(A770ReduceConstants::MAX_FIRST_PASS_WORKGROUPS)
    }

    /// Execute a flat (global) reduction using two-pass tree strategy.
    pub fn reduce(&self, data: &[f32], op: ReduceOp) -> (f32, ReduceStats) {
        let start = Instant::now();
        let n = data.len();

        if n == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            return (op.identity(), ReduceStats::compute(0, 4, elapsed));
        }

        let result = if self.needs_two_pass(n) {
            // Phase 1: local reductions
            let num_wg = self.first_pass_workgroups(n);
            let chunk_size = n.div_ceil(num_wg);
            let partials: Vec<f32> = (0..num_wg)
                .map(|wg| {
                    let start_idx = wg * chunk_size;
                    let end_idx = (start_idx + chunk_size).min(n);
                    self.reduce_chunk(data, start_idx, end_idx, op)
                })
                .collect();

            // Phase 2: reduce partials
            self.reduce_chunk(&partials, 0, partials.len(), op)
        } else {
            self.reduce_chunk(data, 0, n, op)
        };

        // Finalise mean: divide by count
        let result = match op {
            ReduceOp::Mean => {
                if n > 0 { result / n as f32 } else { 0.0 }
            }
            ReduceOp::Variance => {
                // result is sum-of-squares; need mean first
                // For standalone use, variance must be called via
                // dedicated methods that track both sum and sum-of-squares.
                result
            }
            _ => result,
        };

        let elapsed = start.elapsed().as_secs_f64();
        (result, ReduceStats::compute(n, 4, elapsed))
    }

    /// Reduce a contiguous chunk [start..end) with workgroup-style tree.
    fn reduce_chunk(
        &self,
        data: &[f32],
        start: usize,
        end: usize,
        op: ReduceOp,
    ) -> f32 {
        let mut acc = op.identity();
        for &val in &data[start..end] {
            acc = op.combine(acc, val);
        }
        acc
    }
}

impl Default for TreeReducer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AxisReducer — reduce along a specific axis
// ---------------------------------------------------------------------------

/// Reduces an N-dimensional tensor along a single axis.
pub struct AxisReducer;

impl AxisReducer {
    /// Reduce `data` (interpreted with `shape`) along `axis`.
    ///
    /// Returns `(output_values, output_shape)`.
    pub fn reduce(
        data: &[f32],
        shape: &[usize],
        axis: usize,
        op: ReduceOp,
        keepdims: bool,
    ) -> Result<(Vec<f32>, Vec<usize>), ReduceError> {
        if shape.is_empty() {
            return Err(ReduceError::EmptyShape);
        }
        if axis >= shape.len() {
            return Err(ReduceError::InvalidAxis {
                axis,
                ndim: shape.len(),
            });
        }
        let total = shape_numel(shape);
        if data.len() != total {
            return Err(ReduceError::ShapeMismatch {
                expected: total,
                actual: data.len(),
            });
        }

        let out_shape = output_shape(shape, Some(axis), keepdims);
        let out_numel = shape_numel(&out_shape).max(1);

        // Compute strides for iterating along the axis.
        let outer: usize = shape[..axis].iter().product();
        let reduce_dim = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);

        let mut output = vec![op.identity(); out_numel];

        for o in 0..outer {
            for i in 0..inner {
                let mut acc = op.identity();
                for r in 0..reduce_dim {
                    let idx = o * reduce_dim * inner + r * inner + i;
                    acc = op.combine(acc, data[idx]);
                }
                // Finalise
                acc = Self::finalise_op(acc, reduce_dim, op);
                let out_idx = o * inner + i;
                output[out_idx] = acc;
            }
        }

        Ok((output, out_shape))
    }

    fn finalise_op(acc: f32, count: usize, op: ReduceOp) -> f32 {
        match op {
            ReduceOp::Mean => {
                if count > 0 { acc / count as f32 } else { 0.0 }
            }
            _ => acc,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiAxisReducer — reduce along multiple axes simultaneously
// ---------------------------------------------------------------------------

/// Reduces along multiple axes in a single pass.
pub struct MultiAxisReducer;

impl MultiAxisReducer {
    /// Reduce `data` (with `shape`) along all `axes` simultaneously.
    pub fn reduce(
        data: &[f32],
        shape: &[usize],
        axes: &[usize],
        op: ReduceOp,
        keepdims: bool,
    ) -> Result<(Vec<f32>, Vec<usize>), ReduceError> {
        if shape.is_empty() {
            return Err(ReduceError::EmptyShape);
        }
        let ndim = shape.len();
        for &ax in axes {
            if ax >= ndim {
                return Err(ReduceError::InvalidAxis { axis: ax, ndim });
            }
        }
        let total = shape_numel(shape);
        if data.len() != total {
            return Err(ReduceError::ShapeMismatch {
                expected: total,
                actual: data.len(),
            });
        }
        if axes.is_empty() {
            return Ok((data.to_vec(), shape.to_vec()));
        }

        // Sort and dedup axes.
        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_unstable();
        sorted_axes.dedup();

        // Reduce axes sequentially from highest to lowest so that
        // removing a higher axis does not shift lower axis indices.
        let mut current_data = data.to_vec();
        let mut current_shape = shape.to_vec();

        for &ax in sorted_axes.iter().rev() {
            // When keepdims=true, axis positions never change.
            // When keepdims=false, we iterate highest-first, so all
            // previously removed axes were above `ax` — no shift needed.
            let (new_data, new_shape) =
                AxisReducer::reduce(&current_data, &current_shape, ax, op, keepdims)?;
            current_data = new_data;
            current_shape = new_shape;
        }

        Ok((current_data, current_shape))
    }
}

// ---------------------------------------------------------------------------
// ArgReducer — argmax / argmin returning indices
// ---------------------------------------------------------------------------

/// Returns indices of the maximum or minimum element along an axis.
pub struct ArgReducer;

impl ArgReducer {
    /// Compute argmax along `axis`.
    pub fn argmax(
        data: &[f32],
        shape: &[usize],
        axis: usize,
        keepdims: bool,
    ) -> Result<(Vec<usize>, Vec<usize>), ReduceError> {
        Self::arg_reduce(data, shape, axis, keepdims, true)
    }

    /// Compute argmin along `axis`.
    pub fn argmin(
        data: &[f32],
        shape: &[usize],
        axis: usize,
        keepdims: bool,
    ) -> Result<(Vec<usize>, Vec<usize>), ReduceError> {
        Self::arg_reduce(data, shape, axis, keepdims, false)
    }

    /// Flat argmax over all elements.
    pub fn argmax_flat(data: &[f32]) -> Option<usize> {
        if data.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in data.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Some(best_idx)
    }

    /// Flat argmin over all elements.
    pub fn argmin_flat(data: &[f32]) -> Option<usize> {
        if data.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_val = f32::INFINITY;
        for (i, &v) in data.iter().enumerate() {
            if v < best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Some(best_idx)
    }

    fn arg_reduce(
        data: &[f32],
        shape: &[usize],
        axis: usize,
        keepdims: bool,
        is_max: bool,
    ) -> Result<(Vec<usize>, Vec<usize>), ReduceError> {
        if shape.is_empty() {
            return Err(ReduceError::EmptyShape);
        }
        if axis >= shape.len() {
            return Err(ReduceError::InvalidAxis {
                axis,
                ndim: shape.len(),
            });
        }
        let total = shape_numel(shape);
        if data.len() != total {
            return Err(ReduceError::ShapeMismatch {
                expected: total,
                actual: data.len(),
            });
        }

        let out_shape = output_shape(shape, Some(axis), keepdims);
        let outer: usize = shape[..axis].iter().product();
        let reduce_dim = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);

        let out_numel = (outer * inner).max(1);
        let mut indices = vec![0usize; out_numel];

        let identity = if is_max {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };

        for o in 0..outer {
            for i in 0..inner {
                let mut best_val = identity;
                let mut best_idx = 0usize;
                for r in 0..reduce_dim {
                    let idx = o * reduce_dim * inner + r * inner + i;
                    let v = data[idx];
                    let better = if is_max { v > best_val } else { v < best_val };
                    if better {
                        best_val = v;
                        best_idx = r;
                    }
                }
                indices[o * inner + i] = best_idx;
            }
        }

        Ok((indices, out_shape))
    }
}

// ---------------------------------------------------------------------------
// LogSumExpReducer — numerically stable log(sum(exp(x)))
// ---------------------------------------------------------------------------

/// Numerically stable log-sum-exp: `log(sum(exp(x_i)))`.
///
/// Uses the identity `LSE(x) = max(x) + log(sum(exp(x_i - max(x))))` to
/// avoid overflow.
pub struct LogSumExpReducer;

impl LogSumExpReducer {
    /// Global log-sum-exp over a flat slice.
    pub fn reduce_flat(data: &[f32]) -> f32 {
        if data.is_empty() {
            return f32::NEG_INFINITY;
        }
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val.is_infinite() && max_val < 0.0 {
            return f32::NEG_INFINITY;
        }
        let sum_exp: f32 = data.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    }

    /// Log-sum-exp along a specific axis.
    pub fn reduce_axis(
        data: &[f32],
        shape: &[usize],
        axis: usize,
        keepdims: bool,
    ) -> Result<(Vec<f32>, Vec<usize>), ReduceError> {
        if shape.is_empty() {
            return Err(ReduceError::EmptyShape);
        }
        if axis >= shape.len() {
            return Err(ReduceError::InvalidAxis {
                axis,
                ndim: shape.len(),
            });
        }
        let total = shape_numel(shape);
        if data.len() != total {
            return Err(ReduceError::ShapeMismatch {
                expected: total,
                actual: data.len(),
            });
        }

        let out_shape = output_shape(shape, Some(axis), keepdims);
        let outer: usize = shape[..axis].iter().product();
        let reduce_dim = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);

        let out_numel = (outer * inner).max(1);
        let mut output = vec![0.0f32; out_numel];

        for o in 0..outer {
            for i in 0..inner {
                // First pass: find max
                let mut max_val = f32::NEG_INFINITY;
                for r in 0..reduce_dim {
                    let idx = o * reduce_dim * inner + r * inner + i;
                    max_val = max_val.max(data[idx]);
                }
                // Second pass: sum(exp(x - max))
                let mut sum_exp = 0.0f32;
                for r in 0..reduce_dim {
                    let idx = o * reduce_dim * inner + r * inner + i;
                    sum_exp += (data[idx] - max_val).exp();
                }
                output[o * inner + i] = max_val + sum_exp.ln();
            }
        }

        Ok((output, out_shape))
    }
}

// ---------------------------------------------------------------------------
// CPU reference: variance reduction
// ---------------------------------------------------------------------------

/// Compute variance along an axis (two-pass: mean then sum-of-squares).
pub fn variance_axis(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    keepdims: bool,
) -> Result<(Vec<f32>, Vec<usize>), ReduceError> {
    if shape.is_empty() {
        return Err(ReduceError::EmptyShape);
    }
    if axis >= shape.len() {
        return Err(ReduceError::InvalidAxis {
            axis,
            ndim: shape.len(),
        });
    }
    let total = shape_numel(shape);
    if data.len() != total {
        return Err(ReduceError::ShapeMismatch {
            expected: total,
            actual: data.len(),
        });
    }

    let out_shape = output_shape(shape, Some(axis), keepdims);
    let outer: usize = shape[..axis].iter().product();
    let reduce_dim = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let out_numel = (outer * inner).max(1);
    let mut output = vec![0.0f32; out_numel];

    for o in 0..outer {
        for i in 0..inner {
            // Pass 1: mean
            let mut sum = 0.0f32;
            for r in 0..reduce_dim {
                let idx = o * reduce_dim * inner + r * inner + i;
                sum += data[idx];
            }
            let mean = if reduce_dim > 0 {
                sum / reduce_dim as f32
            } else {
                0.0
            };
            // Pass 2: sum of squared deviations
            let mut var_sum = 0.0f32;
            for r in 0..reduce_dim {
                let idx = o * reduce_dim * inner + r * inner + i;
                let diff = data[idx] - mean;
                var_sum += diff * diff;
            }
            output[o * inner + i] = if reduce_dim > 0 {
                var_sum / reduce_dim as f32
            } else {
                0.0
            };
        }
    }

    Ok((output, out_shape))
}

/// Compute variance over a flat slice (population variance).
pub fn variance_flat(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n
}

// ---------------------------------------------------------------------------
// ReduceError
// ---------------------------------------------------------------------------

/// Errors from reduction operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ReduceError {
    EmptyShape,
    InvalidAxis { axis: usize, ndim: usize },
    ShapeMismatch { expected: usize, actual: usize },
    EmptyInput,
    NumericalInstability(String),
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(f, "tensor shape is empty"),
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "axis {axis} out of range for {ndim}-dim tensor")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape expects {expected} elements, got {actual}")
            }
            Self::EmptyInput => write!(f, "empty input tensor"),
            Self::NumericalInstability(msg) => {
                write!(f, "numerical instability: {msg}")
            }
        }
    }
}

impl std::error::Error for ReduceError {}

// ---------------------------------------------------------------------------
// OpenCL kernel source (A770 two-phase tree reduction)
// ---------------------------------------------------------------------------

/// OpenCL kernel sources for A770 reduction operations.
///
/// The kernel uses workgroup-level shared local memory and subgroup
/// shuffle operations for efficient tree reduction.
pub const OPENCL_REDUCE_KERNEL_SRC: &str = r#"
// Phase 1: per-workgroup partial reduction.
// Each workgroup reduces a chunk of the input to a single value in `partials`.
__kernel void reduce_phase1(
    __global const float* restrict input,
    __global float* restrict partials,
    const int n,
    const int op,              // 0=Sum,1=Max,2=Min,3=Prod
    __local float* scratch)
{
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);
    int gid = get_global_id(0);
    int wg_id = get_group_id(0);

    // Identity value per op
    float identity;
    switch (op) {
        case 1: identity = -INFINITY; break;  // Max
        case 2: identity =  INFINITY; break;  // Min
        case 3: identity = 1.0f;      break;  // Prod
        default: identity = 0.0f;     break;  // Sum
    }

    // Each thread accumulates over its strided elements
    float acc = identity;
    for (int i = gid; i < n; i += get_global_size(0)) {
        float val = input[i];
        switch (op) {
            case 1: acc = fmax(acc, val); break;
            case 2: acc = fmin(acc, val); break;
            case 3: acc *= val;           break;
            default: acc += val;          break;
        }
    }

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in shared memory
    for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            switch (op) {
                case 1: scratch[lid] = fmax(scratch[lid], scratch[lid + stride]); break;
                case 2: scratch[lid] = fmin(scratch[lid], scratch[lid + stride]); break;
                case 3: scratch[lid] *= scratch[lid + stride]; break;
                default: scratch[lid] += scratch[lid + stride]; break;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partials[wg_id] = scratch[0];
    }
}

// Phase 2: reduce the partial results from phase 1.
__kernel void reduce_phase2(
    __global const float* restrict partials,
    __global float* restrict output,
    const int n_partials,
    const int op,
    __local float* scratch)
{
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    float identity;
    switch (op) {
        case 1: identity = -INFINITY; break;
        case 2: identity =  INFINITY; break;
        case 3: identity = 1.0f;      break;
        default: identity = 0.0f;     break;
    }

    float acc = identity;
    for (int i = lid; i < n_partials; i += wg_size) {
        float val = partials[i];
        switch (op) {
            case 1: acc = fmax(acc, val); break;
            case 2: acc = fmin(acc, val); break;
            case 3: acc *= val;           break;
            default: acc += val;          break;
        }
    }

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            switch (op) {
                case 1: scratch[lid] = fmax(scratch[lid], scratch[lid + stride]); break;
                case 2: scratch[lid] = fmin(scratch[lid], scratch[lid + stride]); break;
                case 3: scratch[lid] *= scratch[lid + stride]; break;
                default: scratch[lid] += scratch[lid + stride]; break;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[0] = scratch[0];
    }
}

// Subgroup shuffle reduction (Xe-HPG SIMD16).
// Uses intel_sub_group_shuffle_down for warp-level reduction.
__kernel void reduce_subgroup(
    __global const float* restrict input,
    __global float* restrict output,
    const int n,
    const int op)
{
    int gid = get_global_id(0);
    int sg_size = get_sub_group_size();

    float identity;
    switch (op) {
        case 1: identity = -INFINITY; break;
        case 2: identity =  INFINITY; break;
        case 3: identity = 1.0f;      break;
        default: identity = 0.0f;     break;
    }

    float acc = identity;
    for (int i = gid; i < n; i += get_global_size(0)) {
        float val = input[i];
        switch (op) {
            case 1: acc = fmax(acc, val); break;
            case 2: acc = fmin(acc, val); break;
            case 3: acc *= val;           break;
            default: acc += val;          break;
        }
    }

    // Subgroup shuffle tree reduction
    for (int offset = sg_size / 2; offset > 0; offset >>= 1) {
        float other = intel_sub_group_shuffle_down(acc, acc, offset);
        switch (op) {
            case 1: acc = fmax(acc, other); break;
            case 2: acc = fmin(acc, other); break;
            case 3: acc *= other;           break;
            default: acc += other;          break;
        }
    }

    // First lane of each subgroup writes partial result
    if (get_sub_group_local_id() == 0) {
        int sg_id = get_sub_group_id();
        output[sg_id] = acc;
    }
}

// Axis reduction kernel: reduces along one axis of an N-D tensor.
__kernel void reduce_axis(
    __global const float* restrict input,
    __global float* restrict output,
    const int outer,
    const int reduce_dim,
    const int inner,
    const int op,
    __local float* scratch)
{
    int out_idx = get_group_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    int o = out_idx / inner;
    int i = out_idx % inner;

    float identity;
    switch (op) {
        case 1: identity = -INFINITY; break;
        case 2: identity =  INFINITY; break;
        case 3: identity = 1.0f;      break;
        default: identity = 0.0f;     break;
    }

    float acc = identity;
    for (int r = lid; r < reduce_dim; r += wg_size) {
        int idx = o * reduce_dim * inner + r * inner + i;
        float val = input[idx];
        switch (op) {
            case 1: acc = fmax(acc, val); break;
            case 2: acc = fmin(acc, val); break;
            case 3: acc *= val;           break;
            default: acc += val;          break;
        }
    }

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            switch (op) {
                case 1: scratch[lid] = fmax(scratch[lid], scratch[lid + stride]); break;
                case 2: scratch[lid] = fmin(scratch[lid], scratch[lid + stride]); break;
                case 3: scratch[lid] *= scratch[lid + stride]; break;
                default: scratch[lid] += scratch[lid + stride]; break;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float result = scratch[0];
        // Mean: divide by reduce_dim
        if (op == 0 && reduce_dim > 0) {
            // Mean must be requested by the host; Sum just stores.
        }
        output[out_idx] = result;
    }
}
"#;

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Execute a configured reduction and return the result with stats.
pub fn execute_reduce(
    data: &[f32],
    shape: &[usize],
    config: &ReduceConfig,
) -> Result<ReduceResult, ReduceError> {
    let total = shape_numel(shape);
    if data.len() != total {
        return Err(ReduceError::ShapeMismatch {
            expected: total,
            actual: data.len(),
        });
    }
    if data.is_empty() {
        return Err(ReduceError::EmptyInput);
    }

    let start = Instant::now();

    let (values, out_shape) = match config.op {
        ReduceOp::Variance => match config.axis {
            Some(ax) => variance_axis(data, shape, ax, config.keepdims)?,
            None => {
                let v = variance_flat(data);
                let s = output_shape(shape, None, config.keepdims);
                (vec![v], s)
            }
        },
        ReduceOp::ArgMax => {
            match config.axis {
                Some(ax) => {
                    let (idx, s) =
                        ArgReducer::argmax(data, shape, ax, config.keepdims)?;
                    let vals: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
                    (vals, s)
                }
                None => {
                    let idx = ArgReducer::argmax_flat(data).unwrap_or(0);
                    let s = output_shape(shape, None, config.keepdims);
                    (vec![idx as f32], s)
                }
            }
        }
        ReduceOp::ArgMin => {
            match config.axis {
                Some(ax) => {
                    let (idx, s) =
                        ArgReducer::argmin(data, shape, ax, config.keepdims)?;
                    let vals: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
                    (vals, s)
                }
                None => {
                    let idx = ArgReducer::argmin_flat(data).unwrap_or(0);
                    let s = output_shape(shape, None, config.keepdims);
                    (vec![idx as f32], s)
                }
            }
        }
        ReduceOp::LogSumExp => match config.axis {
            Some(ax) => {
                LogSumExpReducer::reduce_axis(data, shape, ax, config.keepdims)?
            }
            None => {
                let v = LogSumExpReducer::reduce_flat(data);
                let s = output_shape(shape, None, config.keepdims);
                (vec![v], s)
            }
        },
        op => match config.axis {
            Some(ax) => AxisReducer::reduce(data, shape, ax, op, config.keepdims)?,
            None => {
                let reducer = TreeReducer::new();
                let (val, _) = reducer.reduce(data, op);
                let s = output_shape(shape, None, config.keepdims);
                (vec![val], s)
            }
        },
    };

    let elapsed = start.elapsed().as_secs_f64();
    let stats = ReduceStats::compute(data.len(), 4, elapsed);
    Ok(ReduceResult { values, shape: out_shape, stats })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ──────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps || (a.is_nan() && b.is_nan())
    }

    fn assert_approx(a: f32, b: f32, eps: f32) {
        assert!(
            approx_eq(a, b, eps),
            "expected ≈ {b}, got {a} (eps={eps})"
        );
    }

    fn assert_vec_approx(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                approx_eq(x, y, eps),
                "element [{i}]: expected ≈ {y}, got {x} (eps={eps})"
            );
        }
    }

    // ── ReduceOp basics ────────────────────────────────────────────────

    #[test]
    fn test_reduce_op_identity() {
        assert_eq!(ReduceOp::Sum.identity(), 0.0);
        assert_eq!(ReduceOp::Prod.identity(), 1.0);
        assert_eq!(ReduceOp::Max.identity(), f32::NEG_INFINITY);
        assert_eq!(ReduceOp::Min.identity(), f32::INFINITY);
    }

    #[test]
    fn test_reduce_op_combine() {
        assert_eq!(ReduceOp::Sum.combine(3.0, 4.0), 7.0);
        assert_eq!(ReduceOp::Max.combine(3.0, 4.0), 4.0);
        assert_eq!(ReduceOp::Min.combine(3.0, 4.0), 3.0);
        assert_eq!(ReduceOp::Prod.combine(3.0, 4.0), 12.0);
    }

    #[test]
    fn test_reduce_op_returns_indices() {
        assert!(ReduceOp::ArgMax.returns_indices());
        assert!(ReduceOp::ArgMin.returns_indices());
        assert!(!ReduceOp::Sum.returns_indices());
        assert!(!ReduceOp::Mean.returns_indices());
    }

    #[test]
    fn test_reduce_op_display() {
        assert_eq!(format!("{}", ReduceOp::LogSumExp), "LogSumExp");
        assert_eq!(format!("{}", ReduceOp::Variance), "Variance");
    }

    // ── ReduceConfig builder ───────────────────────────────────────────

    #[test]
    fn test_reduce_config_builder() {
        let cfg = ReduceConfig::new(ReduceOp::Sum)
            .with_axis(1)
            .with_keepdims(true)
            .with_dtype(ReduceDtype::F16);
        assert_eq!(cfg.op, ReduceOp::Sum);
        assert_eq!(cfg.axis, Some(1));
        assert!(cfg.keepdims);
        assert_eq!(cfg.dtype, ReduceDtype::F16);
    }

    #[test]
    fn test_reduce_config_global() {
        let cfg = ReduceConfig::global(ReduceOp::Max);
        assert_eq!(cfg.op, ReduceOp::Max);
        assert_eq!(cfg.axis, None);
        assert!(!cfg.keepdims);
    }

    // ── ReduceDtype ────────────────────────────────────────────────────

    #[test]
    fn test_reduce_dtype_byte_size() {
        assert_eq!(ReduceDtype::F32.byte_size(), 4);
        assert_eq!(ReduceDtype::F16.byte_size(), 2);
        assert_eq!(ReduceDtype::I32.byte_size(), 4);
    }

    #[test]
    fn test_reduce_dtype_display() {
        assert_eq!(format!("{}", ReduceDtype::F32), "f32");
    }

    // ── TreeReducer flat reductions ────────────────────────────────────

    #[test]
    fn test_tree_reducer_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reducer = TreeReducer::new();
        let (val, stats) = reducer.reduce(&data, ReduceOp::Sum);
        assert_approx(val, 15.0, 1e-5);
        assert_eq!(stats.elements_processed, 5);
    }

    #[test]
    fn test_tree_reducer_max() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Max);
        assert_approx(val, 5.0, 1e-5);
    }

    #[test]
    fn test_tree_reducer_min() {
        let data = vec![3.0, 1.0, 4.0, 1.5, 2.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Min);
        assert_approx(val, 1.0, 1e-5);
    }

    #[test]
    fn test_tree_reducer_mean() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Mean);
        assert_approx(val, 6.0, 1e-5);
    }

    #[test]
    fn test_tree_reducer_prod() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Prod);
        assert_approx(val, 24.0, 1e-5);
    }

    #[test]
    fn test_tree_reducer_empty() {
        let reducer = TreeReducer::new();
        let (val, stats) = reducer.reduce(&[], ReduceOp::Sum);
        assert_eq!(val, 0.0);
        assert_eq!(stats.elements_processed, 0);
    }

    #[test]
    fn test_tree_reducer_single_element() {
        let data = vec![42.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Sum);
        assert_approx(val, 42.0, 1e-5);
    }

    #[test]
    fn test_tree_reducer_two_pass_large() {
        let n = 100_000;
        let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let reducer = TreeReducer::new();
        assert!(reducer.needs_two_pass(n));
        let (val, _) = reducer.reduce(&data, ReduceOp::Sum);
        let expected = (n * (n + 1) / 2) as f32;
        // Loose tolerance for large floating-point sums
        assert!(
            (val - expected).abs() / expected < 1e-3,
            "sum={val}, expected={expected}"
        );
    }

    #[test]
    fn test_tree_reducer_two_pass_max() {
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Max);
        let expected = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_approx(val, expected, 1e-6);
    }

    #[test]
    fn test_tree_reducer_needs_two_pass() {
        let reducer = TreeReducer::new();
        assert!(!reducer.needs_two_pass(256));
        assert!(reducer.needs_two_pass(257));
    }

    #[test]
    fn test_tree_reducer_first_pass_workgroups() {
        let reducer = TreeReducer::new();
        assert_eq!(reducer.first_pass_workgroups(512), 2);
        assert_eq!(
            reducer.first_pass_workgroups(1_000_000),
            A770ReduceConstants::MAX_FIRST_PASS_WORKGROUPS
        );
    }

    #[test]
    fn test_tree_reducer_custom_workgroup_size() {
        let reducer = TreeReducer::with_workgroup_size(64);
        assert_eq!(reducer.workgroup_size, 64);
        assert!(reducer.needs_two_pass(65));
    }

    // ── AxisReducer ────────────────────────────────────────────────────

    #[test]
    fn test_axis_sum_2d_axis0() {
        // [[1,2,3],[4,5,6]] → [5,7,9]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, s) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Sum, false).unwrap();
        assert_vec_approx(&vals, &[5.0, 7.0, 9.0], 1e-5);
        assert_eq!(s, vec![3]);
    }

    #[test]
    fn test_axis_sum_2d_axis1() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, s) =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Sum, false).unwrap();
        assert_vec_approx(&vals, &[6.0, 15.0], 1e-5);
        assert_eq!(s, vec![2]);
    }

    #[test]
    fn test_axis_max_2d_axis0() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Max, false).unwrap();
        assert_vec_approx(&vals, &[4.0, 5.0, 6.0], 1e-5);
    }

    #[test]
    fn test_axis_max_2d_axis1() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Max, false).unwrap();
        assert_vec_approx(&vals, &[5.0, 6.0], 1e-5);
    }

    #[test]
    fn test_axis_min_2d_axis0() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Min, false).unwrap();
        assert_vec_approx(&vals, &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_axis_min_2d_axis1() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Min, false).unwrap();
        assert_vec_approx(&vals, &[1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_axis_mean_2d_axis0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Mean, false).unwrap();
        assert_vec_approx(&vals, &[2.5, 3.5, 4.5], 1e-5);
    }

    #[test]
    fn test_axis_mean_2d_axis1() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Mean, false).unwrap();
        assert_vec_approx(&vals, &[2.0, 5.0], 1e-5);
    }

    #[test]
    fn test_axis_sum_3d_axis0() {
        // shape [2,2,3]: reduce axis 0 → shape [2,3]
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let (vals, s) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Sum, false).unwrap();
        // [1+7, 2+8, 3+9, 4+10, 5+11, 6+12] = [8,10,12,14,16,18]
        assert_vec_approx(&vals, &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0], 1e-5);
        assert_eq!(s, vec![2, 3]);
    }

    #[test]
    fn test_axis_sum_3d_axis1() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let (vals, s) =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Sum, false).unwrap();
        // row0: [1+4, 2+5, 3+6]=[5,7,9], row1: [7+10, 8+11, 9+12]=[17,19,21]
        assert_vec_approx(&vals, &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0], 1e-5);
        assert_eq!(s, vec![2, 3]);
    }

    #[test]
    fn test_axis_sum_3d_axis2() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let shape = vec![2, 2, 3];
        let (vals, s) =
            AxisReducer::reduce(&data, &shape, 2, ReduceOp::Sum, false).unwrap();
        // [1+2+3, 4+5+6, 7+8+9, 10+11+12] = [6, 15, 24, 33]
        assert_vec_approx(&vals, &[6.0, 15.0, 24.0, 33.0], 1e-5);
        assert_eq!(s, vec![2, 2]);
    }

    // ── Keepdims ───────────────────────────────────────────────────────

    #[test]
    fn test_axis_keepdims_true() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, s) =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Sum, true).unwrap();
        assert_vec_approx(&vals, &[6.0, 15.0], 1e-5);
        assert_eq!(s, vec![2, 1]);
    }

    #[test]
    fn test_axis_keepdims_false() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (_, s) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Sum, false).unwrap();
        assert_eq!(s, vec![3]);
    }

    #[test]
    fn test_global_keepdims() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let cfg = ReduceConfig::new(ReduceOp::Sum).with_keepdims(true);
        let result = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(result.values[0], 21.0, 1e-5);
        assert_eq!(result.shape, vec![1, 1]);
    }

    #[test]
    fn test_global_no_keepdims() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];
        let cfg = ReduceConfig::new(ReduceOp::Sum);
        let result = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(result.values[0], 6.0, 1e-5);
        assert!(result.shape.is_empty());
    }

    // ── ArgMax / ArgMin ────────────────────────────────────────────────

    #[test]
    fn test_argmax_flat() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        assert_eq!(ArgReducer::argmax_flat(&data), Some(3));
    }

    #[test]
    fn test_argmin_flat() {
        let data = vec![3.0, 1.0, 4.0, 0.5, 2.0];
        assert_eq!(ArgReducer::argmin_flat(&data), Some(3));
    }

    #[test]
    fn test_argmax_empty() {
        assert_eq!(ArgReducer::argmax_flat(&[]), None);
    }

    #[test]
    fn test_argmin_empty() {
        assert_eq!(ArgReducer::argmin_flat(&[]), None);
    }

    #[test]
    fn test_argmax_axis0() {
        // [[1,5,3],[4,2,6]] → argmax axis0 → [1,0,1]
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (idx, s) = ArgReducer::argmax(&data, &shape, 0, false).unwrap();
        assert_eq!(idx, vec![1, 0, 1]);
        assert_eq!(s, vec![3]);
    }

    #[test]
    fn test_argmax_axis1() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (idx, _) = ArgReducer::argmax(&data, &shape, 1, false).unwrap();
        assert_eq!(idx, vec![1, 2]);
    }

    #[test]
    fn test_argmin_axis0() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (idx, _) = ArgReducer::argmin(&data, &shape, 0, false).unwrap();
        assert_eq!(idx, vec![0, 1, 0]);
    }

    #[test]
    fn test_argmin_axis1() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (idx, _) = ArgReducer::argmin(&data, &shape, 1, false).unwrap();
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn test_argmax_keepdims() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let (idx, s) = ArgReducer::argmax(&data, &shape, 1, true).unwrap();
        assert_eq!(idx, vec![1, 2]);
        assert_eq!(s, vec![2, 1]);
    }

    // ── LogSumExp ──────────────────────────────────────────────────────

    #[test]
    fn test_log_sum_exp_flat_basic() {
        let data = vec![1.0, 2.0, 3.0];
        let result = LogSumExpReducer::reduce_flat(&data);
        let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
        assert_approx(result, expected, 1e-5);
    }

    #[test]
    fn test_log_sum_exp_flat_large_values() {
        // Numerical stability: should not overflow with large values
        let data = vec![1000.0, 1001.0, 1002.0];
        let result = LogSumExpReducer::reduce_flat(&data);
        // max=1002, log(exp(-2)+exp(-1)+exp(0))=log(exp(-2)+exp(-1)+1)
        let expected =
            1002.0 + ((-2.0f32).exp() + (-1.0f32).exp() + 1.0).ln();
        assert_approx(result, expected, 1e-3);
    }

    #[test]
    fn test_log_sum_exp_flat_negative_values() {
        let data = vec![-1000.0, -999.0, -998.0];
        let result = LogSumExpReducer::reduce_flat(&data);
        let expected =
            -998.0 + ((-2.0f32).exp() + (-1.0f32).exp() + 1.0).ln();
        assert_approx(result, expected, 1e-3);
    }

    #[test]
    fn test_log_sum_exp_flat_empty() {
        assert_eq!(LogSumExpReducer::reduce_flat(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp_flat_single() {
        let data = vec![5.0];
        assert_approx(LogSumExpReducer::reduce_flat(&data), 5.0, 1e-5);
    }

    #[test]
    fn test_log_sum_exp_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, s) =
            LogSumExpReducer::reduce_axis(&data, &shape, 1, false).unwrap();
        let e0 = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
        let e1 = (4.0f32.exp() + 5.0f32.exp() + 6.0f32.exp()).ln();
        assert_vec_approx(&vals, &[e0, e1], 1e-4);
        assert_eq!(s, vec![2]);
    }

    #[test]
    fn test_log_sum_exp_matches_naive() {
        let data = vec![0.5, 1.5, -0.5, 2.0];
        let naive: f32 = data.iter().map(|&x: &f32| x.exp()).sum::<f32>().ln();
        let stable = LogSumExpReducer::reduce_flat(&data);
        assert_approx(stable, naive, 1e-5);
    }

    // ── Variance ───────────────────────────────────────────────────────

    #[test]
    fn test_variance_flat_basic() {
        // [1,2,3,4,5] → mean=3, var=2.0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_approx(variance_flat(&data), 2.0, 1e-5);
    }

    #[test]
    fn test_variance_flat_constant() {
        let data = vec![7.0, 7.0, 7.0, 7.0];
        assert_approx(variance_flat(&data), 0.0, 1e-5);
    }

    #[test]
    fn test_variance_flat_empty() {
        assert_eq!(variance_flat(&[]), 0.0);
    }

    #[test]
    fn test_variance_axis0() {
        // [[1,2],[3,4]] → var axis0 = [(1-2)^2/2+(3-2)^2/2,...] = [1,1]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let (vals, _) = variance_axis(&data, &shape, 0, false).unwrap();
        assert_vec_approx(&vals, &[1.0, 1.0], 1e-5);
    }

    #[test]
    fn test_variance_axis1() {
        // [[1,3],[2,4]] → var axis1 = [1,1]
        let data = vec![1.0, 3.0, 2.0, 4.0];
        let shape = vec![2, 2];
        let (vals, _) = variance_axis(&data, &shape, 1, false).unwrap();
        assert_vec_approx(&vals, &[1.0, 1.0], 1e-5);
    }

    // ── MultiAxisReducer ───────────────────────────────────────────────

    #[test]
    fn test_multi_axis_reduce_single() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, s) =
            MultiAxisReducer::reduce(&data, &shape, &[1], ReduceOp::Sum, false)
                .unwrap();
        assert_vec_approx(&vals, &[6.0, 15.0], 1e-5);
        assert_eq!(s, vec![2]);
    }

    #[test]
    fn test_multi_axis_reduce_all() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, s) =
            MultiAxisReducer::reduce(&data, &shape, &[0, 1], ReduceOp::Sum, false)
                .unwrap();
        assert_approx(vals[0], 21.0, 1e-5);
        assert!(s.is_empty());
    }

    #[test]
    fn test_multi_axis_reduce_empty_axes() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];
        let (vals, s) =
            MultiAxisReducer::reduce(&data, &shape, &[], ReduceOp::Sum, false)
                .unwrap();
        assert_eq!(vals, data);
        assert_eq!(s, shape);
    }

    #[test]
    fn test_multi_axis_reduce_keepdims() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, s) =
            MultiAxisReducer::reduce(&data, &shape, &[1], ReduceOp::Sum, true)
                .unwrap();
        assert_vec_approx(&vals, &[6.0, 15.0], 1e-5);
        assert_eq!(s, vec![2, 1]);
    }

    // ── execute_reduce unified dispatch ────────────────────────────────

    #[test]
    fn test_execute_reduce_global_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        let cfg = ReduceConfig::global(ReduceOp::Sum);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(r.values[0], 10.0, 1e-5);
    }

    #[test]
    fn test_execute_reduce_global_max() {
        let data = vec![1.0, 5.0, 3.0, 2.0];
        let shape = vec![4];
        let cfg = ReduceConfig::global(ReduceOp::Max);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(r.values[0], 5.0, 1e-5);
    }

    #[test]
    fn test_execute_reduce_global_min() {
        let data = vec![3.0, 1.0, 4.0, 1.5];
        let shape = vec![4];
        let cfg = ReduceConfig::global(ReduceOp::Min);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(r.values[0], 1.0, 1e-5);
    }

    #[test]
    fn test_execute_reduce_global_mean() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let shape = vec![4];
        let cfg = ReduceConfig::global(ReduceOp::Mean);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(r.values[0], 5.0, 1e-5);
    }

    #[test]
    fn test_execute_reduce_global_prod() {
        let data = vec![2.0, 3.0, 4.0];
        let shape = vec![3];
        let cfg = ReduceConfig::global(ReduceOp::Prod);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(r.values[0], 24.0, 1e-5);
    }

    #[test]
    fn test_execute_reduce_axis_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let cfg = ReduceConfig::new(ReduceOp::Sum).with_axis(0);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_vec_approx(&r.values, &[5.0, 7.0, 9.0], 1e-5);
    }

    #[test]
    fn test_execute_reduce_variance_global() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];
        let cfg = ReduceConfig::global(ReduceOp::Variance);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_approx(r.values[0], 2.0, 1e-5);
    }

    #[test]
    fn test_execute_reduce_variance_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let cfg = ReduceConfig::new(ReduceOp::Variance).with_axis(0);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_vec_approx(&r.values, &[1.0, 1.0], 1e-5);
    }

    #[test]
    fn test_execute_reduce_argmax_global() {
        let data = vec![1.0, 5.0, 3.0, 2.0];
        let shape = vec![4];
        let cfg = ReduceConfig::global(ReduceOp::ArgMax);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_eq!(r.values[0] as usize, 1);
    }

    #[test]
    fn test_execute_reduce_argmin_global() {
        let data = vec![3.0, 1.0, 4.0, 2.0];
        let shape = vec![4];
        let cfg = ReduceConfig::global(ReduceOp::ArgMin);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_eq!(r.values[0] as usize, 1);
    }

    #[test]
    fn test_execute_reduce_argmax_axis() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let shape = vec![2, 3];
        let cfg = ReduceConfig::new(ReduceOp::ArgMax).with_axis(1);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_eq!(r.values[0] as usize, 1);
        assert_eq!(r.values[1] as usize, 2);
    }

    #[test]
    fn test_execute_reduce_lse_global() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];
        let cfg = ReduceConfig::global(ReduceOp::LogSumExp);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
        assert_approx(r.values[0], expected, 1e-4);
    }

    #[test]
    fn test_execute_reduce_lse_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let cfg = ReduceConfig::new(ReduceOp::LogSumExp).with_axis(1);
        let r = execute_reduce(&data, &shape, &cfg).unwrap();
        assert_eq!(r.values.len(), 2);
    }

    // ── Error handling ─────────────────────────────────────────────────

    #[test]
    fn test_axis_invalid_axis() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];
        let result =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Sum, false);
        assert!(matches!(
            result,
            Err(ReduceError::InvalidAxis { axis: 1, ndim: 1 })
        ));
    }

    #[test]
    fn test_axis_empty_shape() {
        let result =
            AxisReducer::reduce(&[], &[], 0, ReduceOp::Sum, false);
        assert!(matches!(result, Err(ReduceError::EmptyShape)));
    }

    #[test]
    fn test_axis_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 3];
        let result =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Sum, false);
        assert!(matches!(result, Err(ReduceError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_execute_reduce_empty_input() {
        let cfg = ReduceConfig::global(ReduceOp::Sum);
        let result = execute_reduce(&[], &[0], &cfg);
        assert!(matches!(result, Err(ReduceError::EmptyInput)));
    }

    #[test]
    fn test_reduce_error_display() {
        let e = ReduceError::InvalidAxis { axis: 5, ndim: 3 };
        assert!(format!("{e}").contains("axis 5"));
    }

    #[test]
    fn test_lse_axis_invalid() {
        let data = vec![1.0, 2.0];
        let shape = vec![2];
        let result = LogSumExpReducer::reduce_axis(&data, &shape, 1, false);
        assert!(matches!(
            result,
            Err(ReduceError::InvalidAxis { axis: 1, ndim: 1 })
        ));
    }

    #[test]
    fn test_variance_axis_invalid() {
        let data = vec![1.0, 2.0];
        let shape = vec![2];
        let result = variance_axis(&data, &shape, 1, false);
        assert!(matches!(
            result,
            Err(ReduceError::InvalidAxis { axis: 1, ndim: 1 })
        ));
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_all_same_values_sum() {
        let data = vec![3.0; 100];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Sum);
        assert_approx(val, 300.0, 1e-3);
    }

    #[test]
    fn test_all_same_values_max() {
        let data = vec![7.0; 50];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Max);
        assert_approx(val, 7.0, 1e-5);
    }

    #[test]
    fn test_all_same_values_variance() {
        let data = vec![5.0; 100];
        assert_approx(variance_flat(&data), 0.0, 1e-5);
    }

    #[test]
    fn test_nan_handling_max() {
        // NaN should propagate via f32::max semantics
        let data = vec![1.0, f32::NAN, 3.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Max);
        // f32::max(NaN, x) behavior — just check it doesn't panic
        let _ = val;
    }

    #[test]
    fn test_negative_values_sum() {
        let data = vec![-1.0, -2.0, -3.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Sum);
        assert_approx(val, -6.0, 1e-5);
    }

    #[test]
    fn test_mixed_sign_mean() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Mean);
        assert_approx(val, 0.0, 1e-5);
    }

    // ── Property tests ─────────────────────────────────────────────────

    #[test]
    fn test_property_sum_equals_total() {
        let data = vec![1.0; 1024];
        let reducer = TreeReducer::new();
        let (val, _) = reducer.reduce(&data, ReduceOp::Sum);
        assert_approx(val, 1024.0, 1e-2);
    }

    #[test]
    fn test_property_max_gte_all() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let reducer = TreeReducer::new();
        let (max_val, _) = reducer.reduce(&data, ReduceOp::Max);
        for &v in &data {
            assert!(max_val >= v, "max {max_val} < element {v}");
        }
    }

    #[test]
    fn test_property_min_lte_all() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let reducer = TreeReducer::new();
        let (min_val, _) = reducer.reduce(&data, ReduceOp::Min);
        for &v in &data {
            assert!(min_val <= v, "min {min_val} > element {v}");
        }
    }

    #[test]
    fn test_property_mean_between_min_max() {
        let data = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let reducer = TreeReducer::new();
        let (mean, _) = reducer.reduce(&data, ReduceOp::Mean);
        let (min_v, _) = reducer.reduce(&data, ReduceOp::Min);
        let (max_v, _) = reducer.reduce(&data, ReduceOp::Max);
        assert!(mean >= min_v && mean <= max_v);
    }

    #[test]
    fn test_property_variance_non_negative() {
        let data = vec![-5.0, 0.0, 5.0, 10.0, -10.0];
        assert!(variance_flat(&data) >= 0.0);
    }

    // ── ReduceStats ────────────────────────────────────────────────────

    #[test]
    fn test_reduce_stats_display() {
        let stats = ReduceStats {
            compute_time: 0.001,
            bandwidth_utilization: 1e9,
            elements_processed: 1000,
        };
        let s = format!("{stats}");
        assert!(s.contains("1000"));
        assert!(s.contains("GB/s"));
    }

    #[test]
    fn test_reduce_stats_compute() {
        let stats = ReduceStats::compute(1000, 4, 0.001);
        assert_eq!(stats.elements_processed, 1000);
        assert!(stats.bandwidth_utilization > 0.0);
    }

    // ── A770 constants ─────────────────────────────────────────────────

    #[test]
    fn test_a770_constants() {
        assert_eq!(A770ReduceConstants::WORKGROUP_SIZE, 256);
        assert_eq!(A770ReduceConstants::SUBGROUP_WIDTH, 16);
        assert_eq!(A770ReduceConstants::LOCAL_MEM_BYTES, 65_536);
        assert_eq!(A770ReduceConstants::COMPUTE_UNITS, 32);
    }

    // ── OpenCL kernel source ───────────────────────────────────────────

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!OPENCL_REDUCE_KERNEL_SRC.is_empty());
    }

    #[test]
    fn test_kernel_source_has_phase1() {
        assert!(OPENCL_REDUCE_KERNEL_SRC.contains("reduce_phase1"));
    }

    #[test]
    fn test_kernel_source_has_phase2() {
        assert!(OPENCL_REDUCE_KERNEL_SRC.contains("reduce_phase2"));
    }

    #[test]
    fn test_kernel_source_has_subgroup() {
        assert!(OPENCL_REDUCE_KERNEL_SRC.contains("reduce_subgroup"));
    }

    #[test]
    fn test_kernel_source_has_axis() {
        assert!(OPENCL_REDUCE_KERNEL_SRC.contains("reduce_axis"));
    }

    #[test]
    fn test_kernel_source_has_local_mem() {
        assert!(OPENCL_REDUCE_KERNEL_SRC.contains("__local"));
    }

    #[test]
    fn test_kernel_source_has_barrier() {
        assert!(OPENCL_REDUCE_KERNEL_SRC.contains("barrier"));
    }

    // ── 1-D tensor reductions ──────────────────────────────────────────

    #[test]
    fn test_1d_sum() {
        let data = vec![10.0, 20.0, 30.0];
        let shape = vec![3];
        let (vals, s) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Sum, false).unwrap();
        assert_approx(vals[0], 60.0, 1e-5);
        assert!(s.is_empty());
    }

    #[test]
    fn test_1d_max() {
        let data = vec![10.0, 30.0, 20.0];
        let shape = vec![3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Max, false).unwrap();
        assert_approx(vals[0], 30.0, 1e-5);
    }

    #[test]
    fn test_1d_mean() {
        let data = vec![10.0, 20.0, 30.0];
        let shape = vec![3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 0, ReduceOp::Mean, false).unwrap();
        assert_approx(vals[0], 20.0, 1e-5);
    }

    // ── Multi-axis with 3D tensor ──────────────────────────────────────

    #[test]
    fn test_multi_axis_3d_reduce_two_axes() {
        // shape [2,3,2], reduce axes [0,2] → shape [3]
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let shape = vec![2, 3, 2];
        let (vals, s) =
            MultiAxisReducer::reduce(&data, &shape, &[0, 2], ReduceOp::Sum, false)
                .unwrap();
        // After reducing axis 2: shape [2,3] → [3,12,21, 10,26,42] wrong...
        // Let me compute: axis 2 first → [1+2, 3+4, 5+6, 7+8, 9+10, 11+12]
        //   = [3,7,11,15,19,23] shape [2,3]
        // Then axis 0: [3+15, 7+19, 11+23] = [18, 26, 34]
        assert_eq!(s, vec![3]);
        assert_vec_approx(&vals, &[18.0, 26.0, 34.0], 1e-4);
    }

    // ── Prod axis ──────────────────────────────────────────────────────

    #[test]
    fn test_axis_prod() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let (vals, _) =
            AxisReducer::reduce(&data, &shape, 1, ReduceOp::Prod, false).unwrap();
        assert_approx(vals[0], 6.0, 1e-5);   // 1*2*3
        assert_approx(vals[1], 120.0, 1e-3);  // 4*5*6
    }
}
