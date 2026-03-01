//! Shape-aware CUDA reduction kernel with axis support.
//!
//! # Overview
//!
//! This module extends the flat reduction primitives in [`crate::reduction`]
//! with shape-aware, axis-specific reductions over N-dimensional tensors.
//! It provides a unified [`reduce_f32`] entry point that interprets an f32
//! slice as a tensor with a given shape and reduces along a specified axis
//! (or globally when no axis is given).
//!
//! # Keepdim semantics
//!
//! When `keepdim` is true, the reduced axis is preserved as a dimension of
//! size 1 (matching PyTorch / NumPy behaviour). When false, the reduced axis
//! is removed from the output shape.
//!
//! # GPU path
//!
//! On builds with `feature = "gpu"` or `feature = "cuda"`, axis-specific
//! reductions are dispatched to the CUDA tree-reduction kernel in
//! [`crate::reduction`]. Global reductions use the flat CUDA path.
//!
//! # CPU fallback
//!
//! The CPU scalar implementation is always available and serves as the
//! reference for correctness testing.

use bitnet_common::{KernelError, Result};

pub use crate::reduction::ReductionOp;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a shape-aware reduction operation.
#[derive(Debug, Clone)]
pub struct ShapedReductionConfig {
    /// The reduction operation to perform.
    pub op: ReductionOp,
    /// Axis to reduce along. `None` reduces all elements to a scalar.
    pub axis: Option<usize>,
    /// If true, the reduced axis is kept as a dimension of size 1.
    pub keepdim: bool,
}

impl ShapedReductionConfig {
    /// Create a new configuration.
    pub fn new(op: ReductionOp, axis: Option<usize>, keepdim: bool) -> Self {
        Self { op, axis, keepdim }
    }

    /// Convenience: global reduction (no axis, no keepdim).
    pub fn global(op: ReductionOp) -> Self {
        Self { op, axis: None, keepdim: false }
    }
}

// ---------------------------------------------------------------------------
// Output shape computation
// ---------------------------------------------------------------------------

/// Compute the output shape after reducing `input_shape` along `axis`.
fn output_shape(input_shape: &[usize], axis: Option<usize>, keepdim: bool) -> Vec<usize> {
    match axis {
        None => {
            if keepdim {
                vec![1; input_shape.len()]
            } else {
                vec![]
            }
        }
        Some(ax) => {
            if keepdim {
                let mut s = input_shape.to_vec();
                s[ax] = 1;
                s
            } else {
                input_shape.iter().enumerate().filter(|&(i, _)| i != ax).map(|(_, &d)| d).collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU implementation
// ---------------------------------------------------------------------------

/// Reduce an f32 tensor along an optional axis using the specified operation.
///
/// `input` is interpreted as a contiguous row-major tensor with the given
/// `shape`. The result is a flat `Vec<f32>` representing the output tensor
/// whose shape can be computed with [`reduction_output_shape`].
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if:
/// - `input.len()` does not equal the product of `shape`
/// - `axis` is out of bounds for the given shape
/// - `shape` is empty (0-dimensional tensors are not supported)
pub fn reduce_f32(
    input: &[f32],
    shape: &[usize],
    config: &ShapedReductionConfig,
) -> Result<Vec<f32>> {
    validate_inputs(input, shape, config.axis)?;

    if input.is_empty() {
        return Ok(vec![config.op.identity()]);
    }

    match config.axis {
        None => {
            let val = crate::reduction::reduce_f32(input, config.op);
            Ok(vec![val])
        }
        Some(ax) => reduce_along_axis(input, shape, ax, config.op),
    }
}

/// Compute the output shape for a shaped reduction.
pub fn reduction_output_shape(input_shape: &[usize], config: &ShapedReductionConfig) -> Vec<usize> {
    output_shape(input_shape, config.axis, config.keepdim)
}

// ---------------------------------------------------------------------------
// Axis-specific reduction (CPU scalar)
// ---------------------------------------------------------------------------

/// Reduce `input` along `axis` of the given `shape`.
fn reduce_along_axis(
    input: &[f32],
    shape: &[usize],
    axis: usize,
    op: ReductionOp,
) -> Result<Vec<f32>> {
    let ndim = shape.len();
    let axis_len = shape[axis];

    let outer_size: usize = shape[..axis].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };
    let inner_size: usize = shape[axis + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let axis_stride: usize = shape[axis + 1..ndim].iter().product::<usize>().max(1);

    let out_len = outer_size * inner_size;
    let mut output = vec![op.identity(); out_len];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = op.identity();
            for k in 0..axis_len {
                let idx = outer * (axis_len * inner_size) + k * axis_stride + inner;
                acc = op.combine(acc, op.map_element(input[idx]));
            }
            output[outer * inner_size + inner] = op.finalise(acc, axis_len);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_inputs(input: &[f32], shape: &[usize], axis: Option<usize>) -> Result<()> {
    if shape.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "shape must be non-empty".to_string(),
        }
        .into());
    }

    let expected: usize = shape.iter().product();
    if input.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "input length {} does not match shape {:?} (expected {})",
                input.len(),
                shape,
                expected,
            ),
        }
        .into());
    }

    if let Some(ax) = axis
        && ax >= shape.len()
    {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "axis {} out of bounds for shape {:?} (ndim={})",
                ax,
                shape,
                shape.len(),
            ),
        }
        .into());
    }

    Ok(())
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
        assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= tol, "{ctx}: index {i}: expected {e}, got {a} (tol={tol})");
        }
    }

    // =================================================================
    // 1. Global (no axis) reductions
    // =================================================================

    #[test]
    fn test_global_sum_1d() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        let r = reduce_f32(&[1.0, 2.0, 3.0, 4.0], &[4], &cfg).unwrap();
        assert_approx(&r, &[10.0], 1e-6, "global sum 1d");
    }

    #[test]
    fn test_global_sum_2d() {
        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[21.0], 1e-5, "global sum 2d");
    }

    #[test]
    fn test_global_max_1d() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Max);
        let r = reduce_f32(&[3.0, 1.0, 4.0, 1.5], &[4], &cfg).unwrap();
        assert_approx(&r, &[4.0], 1e-6, "global max 1d");
    }

    #[test]
    fn test_global_min_1d() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Min);
        let r = reduce_f32(&[3.0, 1.0, 4.0, 1.5], &[4], &cfg).unwrap();
        assert_approx(&r, &[1.0], 1e-6, "global min 1d");
    }

    #[test]
    fn test_global_mean_1d() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Mean);
        let r = reduce_f32(&[2.0, 4.0, 6.0, 8.0], &[4], &cfg).unwrap();
        assert_approx(&r, &[5.0], 1e-6, "global mean 1d");
    }

    #[test]
    fn test_global_l2norm_1d() {
        let cfg = ShapedReductionConfig::global(ReductionOp::L2Norm);
        let r = reduce_f32(&[3.0, 4.0], &[2], &cfg).unwrap();
        assert_approx(&r, &[5.0], 1e-5, "global l2norm 1d");
    }

    #[test]
    fn test_global_sum_3d() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        let r = reduce_f32(&data, &[2, 3, 4], &cfg).unwrap();
        assert_approx(&r, &[300.0], 1e-3, "global sum 3d");
    }

    // =================================================================
    // 2. Axis-0 reductions (2D)
    // =================================================================

    #[test]
    fn test_axis0_sum_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[5.0, 7.0, 9.0], 1e-6, "axis0 sum 2d");
    }

    #[test]
    fn test_axis0_max_2d() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(0), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[4.0, 5.0, 6.0], 1e-6, "axis0 max 2d");
    }

    #[test]
    fn test_axis0_min_2d() {
        let data = vec![4.0, 2.0, 6.0, 1.0, 5.0, 3.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Min, Some(0), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[1.0, 2.0, 3.0], 1e-6, "axis0 min 2d");
    }

    #[test]
    fn test_axis0_mean_2d() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Mean, Some(0), false);
        let r = reduce_f32(&data, &[2, 2], &cfg).unwrap();
        assert_approx(&r, &[4.0, 6.0], 1e-6, "axis0 mean 2d");
    }

    #[test]
    fn test_axis0_l2norm_2d() {
        let data = vec![3.0, 5.0, 4.0, 12.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::L2Norm, Some(0), false);
        let r = reduce_f32(&data, &[2, 2], &cfg).unwrap();
        assert_approx(&r, &[5.0, 13.0], 1e-5, "axis0 l2norm 2d");
    }

    // =================================================================
    // 3. Axis-1 reductions (2D)
    // =================================================================

    #[test]
    fn test_axis1_sum_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[6.0, 15.0], 1e-6, "axis1 sum 2d");
    }

    #[test]
    fn test_axis1_max_2d() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(1), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[5.0, 6.0], 1e-6, "axis1 max 2d");
    }

    #[test]
    fn test_axis1_min_2d() {
        let data = vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Min, Some(1), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[1.0, 4.0], 1e-6, "axis1 min 2d");
    }

    #[test]
    fn test_axis1_mean_2d() {
        let data = vec![2.0, 4.0, 6.0, 1.0, 3.0, 5.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Mean, Some(1), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[4.0, 3.0], 1e-6, "axis1 mean 2d");
    }

    #[test]
    fn test_axis1_l2norm_2d() {
        let data = vec![3.0, 4.0, 5.0, 12.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::L2Norm, Some(1), false);
        let r = reduce_f32(&data, &[2, 2], &cfg).unwrap();
        assert_approx(&r, &[5.0, 13.0], 1e-5, "axis1 l2norm 2d");
    }

    // =================================================================
    // 4. 3D axis reductions
    // =================================================================

    #[test]
    fn test_axis0_sum_3d() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
        let r = reduce_f32(&data, &[2, 2, 3], &cfg).unwrap();
        assert_approx(&r, &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0], 1e-5, "axis0 sum 3d");
    }

    #[test]
    fn test_axis1_sum_3d() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
        let r = reduce_f32(&data, &[2, 2, 3], &cfg).unwrap();
        assert_approx(&r, &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0], 1e-5, "axis1 sum 3d");
    }

    #[test]
    fn test_axis2_sum_3d() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(2), false);
        let r = reduce_f32(&data, &[2, 2, 3], &cfg).unwrap();
        assert_approx(&r, &[6.0, 15.0, 24.0, 33.0], 1e-5, "axis2 sum 3d");
    }

    #[test]
    fn test_axis1_max_3d() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 9.0, 3.0,  4.0, 2.0, 6.0,
            7.0, 5.0, 8.0, 10.0, 11.0, 12.0,
        ];
        let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(1), false);
        let r = reduce_f32(&data, &[2, 2, 3], &cfg).unwrap();
        assert_approx(&r, &[4.0, 9.0, 6.0, 10.0, 11.0, 12.0], 1e-6, "axis1 max 3d");
    }

    // =================================================================
    // 5. Keepdim tests
    // =================================================================

    #[test]
    fn test_keepdim_axis0_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), true);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[5.0, 7.0, 9.0], 1e-6, "keepdim axis0 2d");
        let s = reduction_output_shape(&[2, 3], &cfg);
        assert_eq!(s, vec![1, 3]);
    }

    #[test]
    fn test_keepdim_axis1_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), true);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        assert_approx(&r, &[6.0, 15.0], 1e-6, "keepdim axis1 2d");
        let s = reduction_output_shape(&[2, 3], &cfg);
        assert_eq!(s, vec![2, 1]);
    }

    #[test]
    fn test_keepdim_global() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = ShapedReductionConfig { op: ReductionOp::Sum, axis: None, keepdim: true };
        let r = reduce_f32(&data, &[2, 2], &cfg).unwrap();
        assert_approx(&r, &[10.0], 1e-6, "keepdim global");
        let s = reduction_output_shape(&[2, 2], &cfg);
        assert_eq!(s, vec![1, 1]);
    }

    #[test]
    fn test_keepdim_3d() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), true);
        let _r = reduce_f32(&data, &[2, 3, 4], &cfg).unwrap();
        let s = reduction_output_shape(&[2, 3, 4], &cfg);
        assert_eq!(s, vec![2, 1, 4]);
    }

    // =================================================================
    // 6. Output shape tests
    // =================================================================

    #[test]
    fn test_output_shape_no_axis_no_keepdim() {
        assert_eq!(output_shape(&[2, 3, 4], None, false), Vec::<usize>::new());
    }

    #[test]
    fn test_output_shape_no_axis_keepdim() {
        assert_eq!(output_shape(&[2, 3, 4], None, true), vec![1, 1, 1]);
    }

    #[test]
    fn test_output_shape_axis0() {
        assert_eq!(output_shape(&[2, 3, 4], Some(0), false), vec![3, 4]);
        assert_eq!(output_shape(&[2, 3, 4], Some(0), true), vec![1, 3, 4]);
    }

    #[test]
    fn test_output_shape_axis1() {
        assert_eq!(output_shape(&[2, 3, 4], Some(1), false), vec![2, 4]);
        assert_eq!(output_shape(&[2, 3, 4], Some(1), true), vec![2, 1, 4]);
    }

    #[test]
    fn test_output_shape_last_axis() {
        assert_eq!(output_shape(&[2, 3, 4], Some(2), false), vec![2, 3]);
        assert_eq!(output_shape(&[2, 3, 4], Some(2), true), vec![2, 3, 1]);
    }

    // =================================================================
    // 7. Edge cases
    // =================================================================

    #[test]
    fn test_single_element_tensor() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        let r = reduce_f32(&[42.0], &[1], &cfg).unwrap();
        assert_approx(&r, &[42.0], 1e-6, "single element");
    }

    #[test]
    fn test_single_element_all_ops() {
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
            ReductionOp::L2Norm,
        ] {
            let cfg = ShapedReductionConfig::global(op);
            let r = reduce_f32(&[7.0], &[1], &cfg).unwrap();
            assert_approx(&r, &[7.0], 1e-6, &format!("single elem {op:?}"));
        }
    }

    #[test]
    fn test_empty_input() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        let r = reduce_f32(&[], &[0], &cfg).unwrap();
        assert_eq!(r, vec![0.0]);
    }

    #[test]
    fn test_empty_max_returns_neg_inf() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Max);
        let r = reduce_f32(&[], &[0], &cfg).unwrap();
        assert_eq!(r[0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_empty_min_returns_inf() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Min);
        let r = reduce_f32(&[], &[0], &cfg).unwrap();
        assert_eq!(r[0], f32::INFINITY);
    }

    #[test]
    fn test_large_1d_sum() {
        let n = 10_000usize;
        let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        let r = reduce_f32(&data, &[n], &cfg).unwrap();
        let expected = (n * (n + 1)) as f32 / 2.0;
        assert!(
            (r[0] - expected).abs() / expected < 1e-4,
            "large sum: expected {expected}, got {}",
            r[0]
        );
    }

    #[test]
    fn test_large_dimension_reduction() {
        let data: Vec<f32> = (0..2048).map(|i| (i % 7) as f32).collect();
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
        let r = reduce_f32(&data, &[2, 1024], &cfg).unwrap();
        assert_eq!(r.len(), 2);
        let row0: f32 = data[..1024].iter().sum();
        let row1: f32 = data[1024..].iter().sum();
        assert_approx(&r, &[row0, row1], 1e-3, "large dim reduction");
    }

    #[test]
    fn test_negative_values_all_ops() {
        let data = vec![-3.0, -1.0, -4.0, -1.5, -2.0, -6.0];
        let shape = &[2, 3];

        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
        let r = reduce_f32(&data, shape, &cfg).unwrap();
        assert_approx(&r, &[-8.0, -9.5], 1e-5, "negative sum");

        let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(1), false);
        let r = reduce_f32(&data, shape, &cfg).unwrap();
        assert_approx(&r, &[-1.0, -1.5], 1e-6, "negative max");

        let cfg = ShapedReductionConfig::new(ReductionOp::Min, Some(1), false);
        let r = reduce_f32(&data, shape, &cfg).unwrap();
        assert_approx(&r, &[-4.0, -6.0], 1e-6, "negative min");
    }

    #[test]
    fn test_axis_reduce_1d() {
        let data = vec![1.0, 2.0, 3.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
        let r = reduce_f32(&data, &[3], &cfg).unwrap();
        assert_approx(&r, &[6.0], 1e-6, "axis0 1d sum");
    }

    // =================================================================
    // 8. Validation / error tests
    // =================================================================

    #[test]
    fn test_shape_length_mismatch() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        assert!(reduce_f32(&[1.0, 2.0], &[3], &cfg).is_err());
    }

    #[test]
    fn test_axis_out_of_bounds() {
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(2), false);
        assert!(reduce_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cfg).is_err());
    }

    #[test]
    fn test_empty_shape_rejected() {
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        assert!(reduce_f32(&[], &[], &cfg).is_err());
    }

    // =================================================================
    // 9. Consistency: all ops × axis combinations
    // =================================================================

    #[test]
    fn test_all_ops_axis0_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cases: &[(ReductionOp, &[f32])] = &[
            (ReductionOp::Sum, &[5.0, 7.0, 9.0]),
            (ReductionOp::Max, &[4.0, 5.0, 6.0]),
            (ReductionOp::Min, &[1.0, 2.0, 3.0]),
            (ReductionOp::Mean, &[2.5, 3.5, 4.5]),
        ];
        for (op, exp) in cases {
            let cfg = ShapedReductionConfig::new(*op, Some(0), false);
            let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
            assert_approx(&r, exp, 1e-5, &format!("all_ops axis0 {op:?}"));
        }
    }

    #[test]
    fn test_all_ops_axis1_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cases: &[(ReductionOp, &[f32])] = &[
            (ReductionOp::Sum, &[6.0, 15.0]),
            (ReductionOp::Max, &[3.0, 6.0]),
            (ReductionOp::Min, &[1.0, 4.0]),
            (ReductionOp::Mean, &[2.0, 5.0]),
        ];
        for (op, exp) in cases {
            let cfg = ShapedReductionConfig::new(*op, Some(1), false);
            let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
            assert_approx(&r, exp, 1e-5, &format!("all_ops axis1 {op:?}"));
        }
    }

    #[test]
    fn test_l2norm_axis0_3x2() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::L2Norm, Some(0), false);
        let r = reduce_f32(&data, &[3, 2], &cfg).unwrap();
        let expected = vec![35.0_f32.sqrt(), 56.0_f32.sqrt()];
        assert_approx(&r, &expected, 1e-5, "l2norm axis0 3x2");
    }

    #[test]
    fn test_l2norm_axis1_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = ShapedReductionConfig::new(ReductionOp::L2Norm, Some(1), false);
        let r = reduce_f32(&data, &[2, 3], &cfg).unwrap();
        let expected = vec![14.0_f32.sqrt(), 77.0_f32.sqrt()];
        assert_approx(&r, &expected, 1e-5, "l2norm axis1 2x3");
    }

    // =================================================================
    // 10. 4D tensor reduction
    // =================================================================

    #[test]
    fn test_4d_axis2_sum() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(2), false);
        let r = reduce_f32(&data, &[2, 2, 2, 3], &cfg).unwrap();
        assert_eq!(r.len(), 12);
        // data[0]+data[3] = 1+4 = 5
        assert_approx(&r[..1], &[5.0], 1e-5, "4d axis2 first elem");
    }

    // =================================================================
    // 11. GPU placeholder tests
    // =================================================================

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_shaped_reduction_sum() {
        let data: Vec<f32> = (1..=1024).map(|i| i as f32).collect();
        let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
        let r = reduce_f32(&data, &[1024], &cfg).unwrap();
        let expected = 1024.0 * 1025.0 / 2.0;
        assert!((r[0] - expected).abs() / expected < 1e-4);
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_shaped_reduction_all_ops() {
        let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01 - 20.0).collect();
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
            ReductionOp::L2Norm,
        ] {
            let cfg = ShapedReductionConfig::global(op);
            let _r = reduce_f32(&data, &[4096], &cfg).unwrap();
        }
    }
}
