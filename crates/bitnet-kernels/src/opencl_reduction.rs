//! OpenCL reduction kernels for Intel Arc A770 and other OpenCL devices.
//!
//! # Overview
//!
//! Reduction operations are fundamental building blocks for softmax (max + sum),
//! attention scoring (sum), LayerNorm (mean + L2 norm), and token selection
//! (argmax). This module provides:
//!
//! - **CPU reference implementations** — scalar, numerically stable reductions
//!   for correctness testing and non-GPU environments. Kahan summation is used
//!   where accumulation order matters.
//! - **Axis-aware reductions** — reduce along an arbitrary axis of an N-D tensor.
//! - **OpenCL kernel source** — tree-based parallel reductions using work-group
//!   local memory, targeting Intel Arc A770 (i.e. Gen12.7 / Xe-HPG) but
//!   portable to any conformant OpenCL 1.2+ device.
//!
//! # Kernels
//!
//! The embedded OpenCL C source ([`REDUCTION_SRC`]) contains five kernels:
//!
//! | Kernel            | Operation           |
//! |-------------------|---------------------|
//! | `reduce_sum`      | Sum of elements     |
//! | `reduce_max`      | Maximum element     |
//! | `reduce_min`      | Minimum element     |
//! | `reduce_argmax`   | Index + value of max|
//! | `reduce_argmin`   | Index + value of min|

use std::fmt;

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Reduction operation enum
// ---------------------------------------------------------------------------

/// Specifies which reduction operation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// Index of the maximum element.
    Argmax,
    /// Index of the minimum element.
    ArgMin,
    /// Sum of all elements.
    Sum,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
    /// Arithmetic mean (sum / count).
    Mean,
    /// L2 norm: `sqrt(Σ x²)`.
    L2Norm,
    /// Log-sum-exp: `log(Σ exp(x))`, computed in a numerically stable way.
    LogSumExp,
}

impl fmt::Display for ReductionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Argmax => write!(f, "argmax"),
            Self::ArgMin => write!(f, "argmin"),
            Self::Sum => write!(f, "sum"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Mean => write!(f, "mean"),
            Self::L2Norm => write!(f, "l2_norm"),
            Self::LogSumExp => write!(f, "log_sum_exp"),
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction result
// ---------------------------------------------------------------------------

/// Result of a reduction operation, carrying both value and optional index.
#[derive(Debug, Clone, PartialEq)]
pub struct ReductionResult {
    /// Scalar result of the reduction (or value at the selected index).
    pub value: f32,
    /// Index of the selected element (meaningful for `Argmax` / `ArgMin`).
    pub index: Option<usize>,
    /// Which operation produced this result.
    pub op: ReductionOp,
}

// ---------------------------------------------------------------------------
// Reduction config
// ---------------------------------------------------------------------------

/// Configuration for a reduction dispatch.
#[derive(Debug, Clone)]
pub struct ReductionConfig {
    /// Axis along which to reduce (`None` = flat reduction over all elements).
    pub axis: Option<usize>,
    /// If `true`, the reduced axis is kept as a size-1 dimension.
    pub keep_dims: bool,
    /// When `true`, use Kahan (compensated) summation for `Sum` / `Mean`.
    pub numeric_precision: bool,
}

impl Default for ReductionConfig {
    fn default() -> Self {
        Self { axis: None, keep_dims: false, numeric_precision: true }
    }
}

// ---------------------------------------------------------------------------
// Reduction errors
// ---------------------------------------------------------------------------

/// Errors specific to reduction operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ReductionError {
    /// Input slice was empty and the operation requires at least one element.
    EmptyInput,
    /// Requested axis is out of bounds for the given shape.
    AxisOutOfBounds {
        /// The axis that was requested.
        axis: usize,
        /// Number of dimensions in the tensor.
        ndim: usize,
    },
    /// Shape is inconsistent with the data length.
    ShapeMismatch {
        /// Product of shape dimensions.
        expected: usize,
        /// Actual data length.
        actual: usize,
    },
}

impl fmt::Display for ReductionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "reduction input is empty"),
            Self::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis {axis} out of bounds for {ndim}-D tensor")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "shape expects {expected} elements but data has {actual}"
                )
            }
        }
    }
}

impl std::error::Error for ReductionError {}

impl From<ReductionError> for bitnet_common::BitNetError {
    fn from(e: ReductionError) -> Self {
        bitnet_common::BitNetError::Kernel(KernelError::InvalidArguments {
            reason: e.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Find the index and value of the maximum element.
///
/// Returns `(index, value)`. For empty slices, returns `(0, NEG_INFINITY)`.
/// NaN values are treated as less than any number.
pub fn cpu_argmax(data: &[f32]) -> (usize, f32) {
    if data.is_empty() {
        return (0, f32::NEG_INFINITY);
    }
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        // Use `>` so that NaN never wins (NaN > x is false).
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

/// Find the index and value of the minimum element.
///
/// Returns `(index, value)`. For empty slices, returns `(0, INFINITY)`.
/// NaN values are treated as greater than any number.
pub fn cpu_argmin(data: &[f32]) -> (usize, f32) {
    if data.is_empty() {
        return (0, f32::INFINITY);
    }
    let mut best_idx = 0;
    let mut best_val = f32::INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

/// Sum with Kahan (compensated) summation for numerical stability.
///
/// Returns `0.0` for an empty slice.
pub fn cpu_sum(data: &[f32]) -> f32 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &v in data {
        let y = v as f64 - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum as f32
}

/// Maximum element. Returns `NEG_INFINITY` for an empty slice.
pub fn cpu_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Minimum element. Returns `INFINITY` for an empty slice.
pub fn cpu_min(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Arithmetic mean using Kahan summation for stability.
///
/// Returns `0.0` for an empty slice.
pub fn cpu_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    cpu_sum(data) / data.len() as f32
}

/// L2 norm: `sqrt(Σ x²)`.
///
/// Returns `0.0` for an empty slice.
pub fn cpu_l2_norm(data: &[f32]) -> f32 {
    let sum_sq: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum();
    sum_sq.sqrt() as f32
}

/// Numerically stable log-sum-exp: `log(Σ exp(xᵢ))`.
///
/// Uses the identity `log Σ exp(xᵢ) = m + log Σ exp(xᵢ − m)` where
/// `m = max(x)` to avoid overflow / underflow.
///
/// Returns `NEG_INFINITY` for an empty slice.
pub fn cpu_log_sum_exp(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    let m = cpu_max(data);
    if m.is_infinite() && m.is_sign_negative() {
        return f32::NEG_INFINITY;
    }
    let sum: f64 = data
        .iter()
        .map(|&v| ((v - m) as f64).exp())
        .sum();
    m + sum.ln() as f32
}

// ---------------------------------------------------------------------------
// Axis reduction (CPU reference)
// ---------------------------------------------------------------------------

/// Reduce `data` along a specific `axis` of a tensor with the given `shape`.
///
/// Returns a flat `Vec<f32>` containing the reduction results. The output
/// length is `product(shape) / shape[axis]`.
pub fn cpu_reduce_axis(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    op: ReductionOp,
) -> Result<Vec<f32>> {
    // Validate inputs.
    if axis >= shape.len() {
        return Err(ReductionError::AxisOutOfBounds {
            axis,
            ndim: shape.len(),
        }
        .into());
    }
    let total: usize = shape.iter().product();
    if total != data.len() {
        return Err(
            ReductionError::ShapeMismatch { expected: total, actual: data.len() }
                .into(),
        );
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return Err(ReductionError::EmptyInput.into());
    }

    // outer_size = product of dims before axis
    let outer: usize = shape[..axis].iter().product();
    // inner_size = product of dims after axis
    let inner: usize = shape[axis + 1..].iter().product();

    let out_len = outer * inner;
    let mut output = vec![0.0_f32; out_len];

    for o in 0..outer {
        for i in 0..inner {
            // Gather elements along the reduction axis.
            let mut lane = Vec::with_capacity(axis_len);
            for a in 0..axis_len {
                let idx = o * axis_len * inner + a * inner + i;
                lane.push(data[idx]);
            }
            let val = reduce_slice(&lane, op);
            output[o * inner + i] = val;
        }
    }

    Ok(output)
}

/// Apply a [`ReductionOp`] to a flat slice and return a scalar.
fn reduce_slice(data: &[f32], op: ReductionOp) -> f32 {
    match op {
        ReductionOp::Argmax => cpu_argmax(data).1,
        ReductionOp::ArgMin => cpu_argmin(data).1,
        ReductionOp::Sum => cpu_sum(data),
        ReductionOp::Max => cpu_max(data),
        ReductionOp::Min => cpu_min(data),
        ReductionOp::Mean => cpu_mean(data),
        ReductionOp::L2Norm => cpu_l2_norm(data),
        ReductionOp::LogSumExp => cpu_log_sum_exp(data),
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C kernels for tree-based parallel reductions.
///
/// Work-group local memory is used for the tree phase. Each work-group
/// produces one partial result; for inputs larger than one work-group the host
/// must launch a second pass (or use CPU finalisation).
///
/// Targets Intel Arc A770 (Xe-HPG, 512 EUs, 64 KiB SLM per sub-slice) but
/// is portable to any OpenCL 1.2+ device.
pub const REDUCTION_SRC: &str = r#"
// ------------------------------------------------------------------
// reduce_sum — parallel sum with local-memory tree reduction
// ------------------------------------------------------------------
__kernel void reduce_sum(
    __global const float* restrict input,
    __global       float* restrict output,
    const int n,
    __local        float* scratch)
{
    const int lid  = get_local_id(0);
    const int gid  = get_global_id(0);
    const int gsz  = get_global_size(0);

    float acc = 0.0f;
    for (int i = gid; i < n; i += gsz)
        acc += input[i];

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        output[get_group_id(0)] = scratch[0];
}

// ------------------------------------------------------------------
// reduce_max — parallel max
// ------------------------------------------------------------------
__kernel void reduce_max(
    __global const float* restrict input,
    __global       float* restrict output,
    const int n,
    __local        float* scratch)
{
    const int lid  = get_local_id(0);
    const int gid  = get_global_id(0);
    const int gsz  = get_global_size(0);

    float acc = -INFINITY;
    for (int i = gid; i < n; i += gsz)
        acc = fmax(acc, input[i]);

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        output[get_group_id(0)] = scratch[0];
}

// ------------------------------------------------------------------
// reduce_min — parallel min
// ------------------------------------------------------------------
__kernel void reduce_min(
    __global const float* restrict input,
    __global       float* restrict output,
    const int n,
    __local        float* scratch)
{
    const int lid  = get_local_id(0);
    const int gid  = get_global_id(0);
    const int gsz  = get_global_size(0);

    float acc = INFINITY;
    for (int i = gid; i < n; i += gsz)
        acc = fmin(acc, input[i]);

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] = fmin(scratch[lid], scratch[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        output[get_group_id(0)] = scratch[0];
}

// ------------------------------------------------------------------
// reduce_argmax — parallel argmax (value + index)
// ------------------------------------------------------------------
__kernel void reduce_argmax(
    __global const float* restrict input,
    __global       float* restrict out_val,
    __global       int*   restrict out_idx,
    const int n,
    __local        float* scratch_val,
    __local        int*   scratch_idx)
{
    const int lid  = get_local_id(0);
    const int gid  = get_global_id(0);
    const int gsz  = get_global_size(0);

    float best_val = -INFINITY;
    int   best_idx = 0;
    for (int i = gid; i < n; i += gsz) {
        float v = input[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    scratch_val[lid] = best_val;
    scratch_idx[lid] = best_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (scratch_val[lid + stride] > scratch_val[lid]) {
                scratch_val[lid] = scratch_val[lid + stride];
                scratch_idx[lid] = scratch_idx[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out_val[get_group_id(0)] = scratch_val[0];
        out_idx[get_group_id(0)] = scratch_idx[0];
    }
}

// ------------------------------------------------------------------
// reduce_argmin — parallel argmin (value + index)
// ------------------------------------------------------------------
__kernel void reduce_argmin(
    __global const float* restrict input,
    __global       float* restrict out_val,
    __global       int*   restrict out_idx,
    const int n,
    __local        float* scratch_val,
    __local        int*   scratch_idx)
{
    const int lid  = get_local_id(0);
    const int gid  = get_global_id(0);
    const int gsz  = get_global_size(0);

    float best_val = INFINITY;
    int   best_idx = 0;
    for (int i = gid; i < n; i += gsz) {
        float v = input[i];
        if (v < best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    scratch_val[lid] = best_val;
    scratch_idx[lid] = best_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (scratch_val[lid + stride] < scratch_val[lid]) {
                scratch_val[lid] = scratch_val[lid + stride];
                scratch_idx[lid] = scratch_idx[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out_val[get_group_id(0)] = scratch_val[0];
        out_idx[get_group_id(0)] = scratch_idx[0];
    }
}
"#;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- argmax -----------------------------------------------------------

    #[test]
    fn test_argmax_basic() {
        let data = [1.0, 3.0, 2.0, 5.0, 4.0];
        let (idx, val) = cpu_argmax(&data);
        assert_eq!(idx, 3);
        assert_eq!(val, 5.0);
    }

    #[test]
    fn test_argmax_single() {
        let (idx, val) = cpu_argmax(&[42.0]);
        assert_eq!(idx, 0);
        assert_eq!(val, 42.0);
    }

    #[test]
    fn test_argmax_empty() {
        let (idx, val) = cpu_argmax(&[]);
        assert_eq!(idx, 0);
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    #[test]
    fn test_argmax_all_same() {
        let data = [7.0; 5];
        let (idx, val) = cpu_argmax(&data);
        assert_eq!(idx, 0); // first occurrence
        assert_eq!(val, 7.0);
    }

    #[test]
    fn test_argmax_with_nan() {
        let data = [1.0, f32::NAN, 3.0];
        let (idx, val) = cpu_argmax(&data);
        // NaN never wins with `>`
        assert_eq!(idx, 2);
        assert_eq!(val, 3.0);
    }

    #[test]
    fn test_argmax_negative_values() {
        let data = [-5.0, -1.0, -3.0];
        let (idx, val) = cpu_argmax(&data);
        assert_eq!(idx, 1);
        assert_eq!(val, -1.0);
    }

    #[test]
    fn test_argmax_valid_index_property() {
        let data = [10.0, 20.0, 30.0, 5.0];
        let (idx, _) = cpu_argmax(&data);
        assert!(idx < data.len(), "argmax index must be valid");
    }

    // -- argmin -----------------------------------------------------------

    #[test]
    fn test_argmin_basic() {
        let data = [3.0, 1.0, 4.0, 0.5, 2.0];
        let (idx, val) = cpu_argmin(&data);
        assert_eq!(idx, 3);
        assert_eq!(val, 0.5);
    }

    #[test]
    fn test_argmin_empty() {
        let (idx, val) = cpu_argmin(&[]);
        assert_eq!(idx, 0);
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_argmin_with_nan() {
        let data = [3.0, f32::NAN, 1.0];
        let (idx, val) = cpu_argmin(&data);
        assert_eq!(idx, 2);
        assert_eq!(val, 1.0);
    }

    // -- sum --------------------------------------------------------------

    #[test]
    fn test_sum_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert!((cpu_sum(&data) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_empty() {
        assert_eq!(cpu_sum(&[]), 0.0);
    }

    #[test]
    fn test_sum_single() {
        assert_eq!(cpu_sum(&[99.0]), 99.0);
    }

    #[test]
    fn test_sum_kahan_accuracy() {
        // One large value followed by many small values. Naive f32
        // summation loses the small contributions; Kahan keeps them.
        let mut data = vec![1e8_f32];
        for _ in 0..10_000 {
            data.push(1.0);
        }
        let result = cpu_sum(&data);
        let expected = 1e8 + 10_000.0;
        let rel_err = ((result - expected) / expected).abs();
        assert!(
            rel_err < 1e-6,
            "Kahan sum relative error {rel_err} exceeds threshold"
        );
    }

    #[test]
    fn test_sum_negative() {
        let data = [-1.0, -2.0, -3.0];
        assert!((cpu_sum(&data) - (-6.0)).abs() < 1e-6);
    }

    // -- max / min --------------------------------------------------------

    #[test]
    fn test_max_basic() {
        assert_eq!(cpu_max(&[1.0, 5.0, 3.0]), 5.0);
    }

    #[test]
    fn test_max_empty() {
        assert!(cpu_max(&[]).is_infinite() && cpu_max(&[]).is_sign_negative());
    }

    #[test]
    fn test_min_basic() {
        assert_eq!(cpu_min(&[4.0, 2.0, 6.0]), 2.0);
    }

    #[test]
    fn test_min_empty() {
        assert!(cpu_min(&[]).is_infinite() && cpu_min(&[]).is_sign_positive());
    }

    #[test]
    fn test_max_with_inf() {
        let data = [1.0, f32::INFINITY, 3.0];
        assert_eq!(cpu_max(&data), f32::INFINITY);
    }

    #[test]
    fn test_min_with_neg_inf() {
        let data = [1.0, f32::NEG_INFINITY, 3.0];
        assert_eq!(cpu_min(&data), f32::NEG_INFINITY);
    }

    // -- mean -------------------------------------------------------------

    #[test]
    fn test_mean_basic() {
        let data = [2.0, 4.0, 6.0];
        assert!((cpu_mean(&data) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(cpu_mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_between_min_and_max() {
        let data = [10.0, 20.0, 30.0, 5.0, 15.0];
        let m = cpu_mean(&data);
        assert!(m >= cpu_min(&data));
        assert!(m <= cpu_max(&data));
    }

    // -- l2_norm ----------------------------------------------------------

    #[test]
    fn test_l2_norm_basic() {
        let data = [3.0, 4.0];
        assert!((cpu_l2_norm(&data) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_empty() {
        assert_eq!(cpu_l2_norm(&[]), 0.0);
    }

    #[test]
    fn test_l2_norm_single() {
        assert!((cpu_l2_norm(&[-7.0]) - 7.0).abs() < 1e-6);
    }

    // -- log_sum_exp ------------------------------------------------------

    #[test]
    fn test_log_sum_exp_basic() {
        let data = [1.0, 2.0, 3.0];
        // log(e^1 + e^2 + e^3)
        let expected =
            (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln() as f32;
        assert!(
            (cpu_log_sum_exp(&data) - expected).abs() < 1e-4,
            "log_sum_exp mismatch"
        );
    }

    #[test]
    fn test_log_sum_exp_empty() {
        assert!(cpu_log_sum_exp(&[]).is_infinite());
    }

    #[test]
    fn test_log_sum_exp_numerical_stability() {
        // Large values that would overflow naive exp.
        let data = [1000.0, 1001.0, 1002.0];
        let result = cpu_log_sum_exp(&data);
        // Should be close to 1002 + log(e^-2 + e^-1 + 1)
        assert!(result.is_finite(), "log_sum_exp must not overflow");
        assert!(
            (result - 1002.0).abs() < 2.0,
            "log_sum_exp result {result} is unreasonably far from 1002"
        );
    }

    #[test]
    fn test_log_sum_exp_single() {
        let result = cpu_log_sum_exp(&[5.0]);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_sum_exp_identical() {
        // log(N * exp(v)) = v + log(N)
        let n = 100;
        let v = 3.0_f32;
        let data = vec![v; n];
        let expected = v + (n as f32).ln();
        assert!(
            (cpu_log_sum_exp(&data) - expected).abs() < 1e-3,
            "log_sum_exp of identical values"
        );
    }

    // -- axis reduction ---------------------------------------------------

    #[test]
    fn test_reduce_axis_2d_sum_axis0() {
        // shape [2, 3]
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = cpu_reduce_axis(&data, &[2, 3], 0, ReductionOp::Sum)
            .expect("reduce_axis failed");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 7.0).abs() < 1e-6);
        assert!((result[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_axis_2d_sum_axis1() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = cpu_reduce_axis(&data, &[2, 3], 1, ReductionOp::Sum)
            .expect("reduce_axis failed");
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_axis_2d_max_axis0() {
        let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let result = cpu_reduce_axis(&data, &[2, 3], 0, ReductionOp::Max)
            .expect("reduce_axis failed");
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reduce_axis_3d_sum_axis1() {
        // shape [2, 3, 2]
        // [[[1,2],[3,4],[5,6]],
        //  [[7,8],[9,10],[11,12]]]
        let data: Vec<f32> =
            (1..=12).map(|x| x as f32).collect();
        let result =
            cpu_reduce_axis(&data, &[2, 3, 2], 1, ReductionOp::Sum)
                .expect("reduce_axis failed");
        // output shape [2, 2]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 9.0).abs() < 1e-6); // 1+3+5
        assert!((result[1] - 12.0).abs() < 1e-6); // 2+4+6
        assert!((result[2] - 27.0).abs() < 1e-6); // 7+9+11
        assert!((result[3] - 30.0).abs() < 1e-6); // 8+10+12
    }

    #[test]
    fn test_reduce_axis_oob() {
        let data = [1.0, 2.0];
        let result = cpu_reduce_axis(&data, &[2], 1, ReductionOp::Sum);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_axis_shape_mismatch() {
        let data = [1.0, 2.0, 3.0];
        let result = cpu_reduce_axis(&data, &[2, 3], 0, ReductionOp::Sum);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_axis_mean() {
        let data = [2.0, 4.0, 6.0, 8.0];
        let result = cpu_reduce_axis(&data, &[2, 2], 0, ReductionOp::Mean)
            .expect("reduce_axis failed");
        // mean along axis 0: [(2+6)/2, (4+8)/2] = [4.0, 6.0]
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
    }

    // -- large array tests ------------------------------------------------

    #[test]
    fn test_sum_large() {
        let data: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.01).collect();
        let result = cpu_sum(&data);
        // sum of 0..9999 = 9999*10000/2 = 49_995_000, * 0.01 = 499_950
        let expected = 499_950.0_f32;
        let rel_err = ((result - expected) / expected).abs();
        assert!(rel_err < 1e-4, "large sum relative error {rel_err}");
    }

    #[test]
    fn test_argmax_large() {
        let mut data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        data[7777] = 999_999.0;
        let (idx, val) = cpu_argmax(&data);
        assert_eq!(idx, 7777);
        assert_eq!(val, 999_999.0);
    }

    #[test]
    fn test_l2_norm_large() {
        let data = vec![1.0_f32; 10_000];
        let result = cpu_l2_norm(&data);
        assert!((result - 100.0).abs() < 0.01); // sqrt(10_000)
    }

    // -- ReductionResult / config / error ---------------------------------

    #[test]
    fn test_reduction_result_display() {
        let r = ReductionResult {
            value: 42.0,
            index: Some(3),
            op: ReductionOp::Argmax,
        };
        assert_eq!(r.op.to_string(), "argmax");
        assert_eq!(r.index, Some(3));
    }

    #[test]
    fn test_reduction_config_default() {
        let cfg = ReductionConfig::default();
        assert!(cfg.axis.is_none());
        assert!(!cfg.keep_dims);
        assert!(cfg.numeric_precision);
    }

    #[test]
    fn test_reduction_error_display() {
        let e = ReductionError::EmptyInput;
        assert_eq!(e.to_string(), "reduction input is empty");

        let e2 = ReductionError::AxisOutOfBounds { axis: 5, ndim: 3 };
        assert!(e2.to_string().contains("5"));
    }

    #[test]
    fn test_reduction_error_into_bitnet() {
        let e = ReductionError::ShapeMismatch { expected: 6, actual: 4 };
        let be: bitnet_common::BitNetError = e.into();
        let msg = format!("{be}");
        assert!(msg.contains("6") || msg.contains("4"));
    }

    // -- OpenCL source sanity ---------------------------------------------

    #[test]
    fn test_opencl_source_contains_kernels() {
        assert!(REDUCTION_SRC.contains("reduce_sum"));
        assert!(REDUCTION_SRC.contains("reduce_max"));
        assert!(REDUCTION_SRC.contains("reduce_min"));
        assert!(REDUCTION_SRC.contains("reduce_argmax"));
        assert!(REDUCTION_SRC.contains("reduce_argmin"));
    }

    #[test]
    fn test_opencl_source_uses_local_memory() {
        assert!(REDUCTION_SRC.contains("__local"));
        assert!(REDUCTION_SRC.contains("barrier(CLK_LOCAL_MEM_FENCE)"));
    }
}
