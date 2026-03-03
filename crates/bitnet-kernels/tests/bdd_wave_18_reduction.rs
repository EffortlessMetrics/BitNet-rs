//! BDD Wave 18 — Reduction operation integration tests.
//!
//! Tests flat, row/column, and shaped reductions across all five
//! [`ReductionOp`] variants: Sum, Max, Min, Mean, L2Norm.

use bitnet_kernels::reduction::{ReductionOp, reduce_cols_f32, reduce_f32, reduce_rows_f32};
use bitnet_kernels::shaped_reduction::{
    ShapedReductionConfig, reduce_f32 as shaped_reduce_f32, reduction_output_shape,
};

const TOL: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
}

// ── Flat reductions ────────────────────────────────────────────────

#[test]
fn given_simple_vector_when_sum_reduced_then_returns_correct_total() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = reduce_f32(&input, ReductionOp::Sum);
    assert!(approx_eq(result, 10.0, TOL), "expected 10.0, got {result}");
}

#[test]
fn given_simple_vector_when_max_reduced_then_returns_largest_element() {
    let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let result = reduce_f32(&input, ReductionOp::Max);
    assert!(approx_eq(result, 5.0, TOL), "expected 5.0, got {result}");
}

#[test]
fn given_simple_vector_when_min_reduced_then_returns_smallest_element() {
    let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let result = reduce_f32(&input, ReductionOp::Min);
    assert!(approx_eq(result, 1.0, TOL), "expected 1.0, got {result}");
}

#[test]
fn given_simple_vector_when_mean_reduced_then_returns_arithmetic_mean() {
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let result = reduce_f32(&input, ReductionOp::Mean);
    assert!(approx_eq(result, 5.0, TOL), "expected 5.0, got {result}");
}

#[test]
fn given_simple_vector_when_l2norm_reduced_then_returns_euclidean_norm() {
    let input = vec![3.0, 4.0];
    let result = reduce_f32(&input, ReductionOp::L2Norm);
    assert!(approx_eq(result, 5.0, TOL), "expected 5.0, got {result}");
}

#[test]
fn given_empty_slice_when_sum_reduced_then_returns_identity_zero() {
    let input: Vec<f32> = vec![];
    let result = reduce_f32(&input, ReductionOp::Sum);
    assert!(approx_eq(result, 0.0, TOL), "sum identity should be 0.0");
}

#[test]
fn given_empty_slice_when_max_reduced_then_returns_neg_infinity() {
    let input: Vec<f32> = vec![];
    let result = reduce_f32(&input, ReductionOp::Max);
    assert!(result == f32::NEG_INFINITY, "max identity should be -inf");
}

#[test]
fn given_single_element_when_any_op_reduced_then_returns_that_element() {
    let input = vec![42.0];
    for op in [ReductionOp::Sum, ReductionOp::Max, ReductionOp::Min, ReductionOp::Mean] {
        let result = reduce_f32(&input, op);
        assert!(approx_eq(result, 42.0, TOL), "single-element {op:?} should be 42.0, got {result}");
    }
}

#[test]
fn given_negative_values_when_sum_reduced_then_handles_sign_correctly() {
    let input = vec![-3.0, -1.0, 2.0, 5.0];
    let result = reduce_f32(&input, ReductionOp::Sum);
    assert!(approx_eq(result, 3.0, TOL));
}

#[test]
fn given_all_zeros_when_l2norm_reduced_then_returns_zero() {
    let input = vec![0.0; 10];
    let result = reduce_f32(&input, ReductionOp::L2Norm);
    assert!(approx_eq(result, 0.0, TOL));
}

// ── Row-wise reductions ────────────────────────────────────────────

#[test]
fn given_2x3_matrix_when_row_sum_then_returns_row_sums() {
    // [[1, 2, 3], [4, 5, 6]]
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Sum).unwrap();
    assert_eq!(result.len(), 2);
    assert!(approx_eq(result[0], 6.0, TOL));
    assert!(approx_eq(result[1], 15.0, TOL));
}

#[test]
fn given_2x3_matrix_when_row_max_then_returns_row_maxima() {
    let matrix = vec![1.0, 9.0, 3.0, 4.0, 2.0, 6.0];
    let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Max).unwrap();
    assert!(approx_eq(result[0], 9.0, TOL));
    assert!(approx_eq(result[1], 6.0, TOL));
}

#[test]
fn given_2x3_matrix_when_row_mean_then_returns_row_means() {
    let matrix = vec![3.0, 6.0, 9.0, 10.0, 20.0, 30.0];
    let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Mean).unwrap();
    assert!(approx_eq(result[0], 6.0, TOL));
    assert!(approx_eq(result[1], 20.0, TOL));
}

// ── Column-wise reductions ─────────────────────────────────────────

#[test]
fn given_3x2_matrix_when_col_sum_then_returns_column_sums() {
    // [[1, 2], [3, 4], [5, 6]]
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = reduce_cols_f32(&matrix, 3, 2, ReductionOp::Sum).unwrap();
    assert_eq!(result.len(), 2);
    assert!(approx_eq(result[0], 9.0, TOL)); // 1+3+5
    assert!(approx_eq(result[1], 12.0, TOL)); // 2+4+6
}

#[test]
fn given_3x2_matrix_when_col_min_then_returns_column_minima() {
    let matrix = vec![5.0, 2.0, 3.0, 8.0, 1.0, 4.0];
    let result = reduce_cols_f32(&matrix, 3, 2, ReductionOp::Min).unwrap();
    assert!(approx_eq(result[0], 1.0, TOL));
    assert!(approx_eq(result[1], 2.0, TOL));
}

// ── Dimension mismatch error handling ──────────────────────────────

#[test]
fn given_mismatched_matrix_size_when_row_reduced_then_returns_error() {
    let matrix = vec![1.0, 2.0, 3.0];
    let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Sum);
    assert!(result.is_err(), "should fail when matrix.len() != rows*cols");
}

#[test]
fn given_zero_rows_when_row_reduced_then_returns_error() {
    let matrix: Vec<f32> = vec![];
    let result = reduce_rows_f32(&matrix, 0, 3, ReductionOp::Sum);
    assert!(result.is_err(), "should fail on zero rows");
}

// ── Shaped reductions ──────────────────────────────────────────────

#[test]
fn given_2d_tensor_when_global_sum_then_returns_single_scalar() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let config = ShapedReductionConfig::global(ReductionOp::Sum);

    let result = shaped_reduce_f32(&input, &shape, &config).unwrap();
    assert_eq!(result.len(), 1);
    assert!(approx_eq(result[0], 21.0, TOL));
}

#[test]
fn given_2d_tensor_when_reduced_along_axis0_then_output_has_cols_elements() {
    // shape [2, 3], reduce axis 0 → output [3]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);

    let result = shaped_reduce_f32(&input, &shape, &config).unwrap();
    assert_eq!(result.len(), 3);
    assert!(approx_eq(result[0], 5.0, TOL)); // 1+4
    assert!(approx_eq(result[1], 7.0, TOL)); // 2+5
    assert!(approx_eq(result[2], 9.0, TOL)); // 3+6
}

#[test]
fn given_2d_tensor_when_reduced_along_axis1_then_output_has_rows_elements() {
    // shape [2, 3], reduce axis 1 → output [2]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);

    let result = shaped_reduce_f32(&input, &shape, &config).unwrap();
    assert_eq!(result.len(), 2);
    assert!(approx_eq(result[0], 6.0, TOL)); // 1+2+3
    assert!(approx_eq(result[1], 15.0, TOL)); // 4+5+6
}

#[test]
fn given_3d_tensor_when_reduced_along_middle_axis_then_shape_is_correct() {
    // shape [2, 3, 4] → reduce axis 1 → output shape [2, 4]
    let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let shape = vec![2, 3, 4];
    let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);

    let result = shaped_reduce_f32(&input, &shape, &config).unwrap();
    // 2 * 4 = 8 elements in output
    assert_eq!(result.len(), 8);
}

#[test]
fn given_2d_tensor_when_keepdim_true_then_output_shape_preserves_axis() {
    let shape = vec![3, 4];
    let config = ShapedReductionConfig::new(ReductionOp::Mean, Some(1), true);
    let out_shape = reduction_output_shape(&shape, &config);
    assert_eq!(out_shape, vec![3, 1]);
}

#[test]
fn given_2d_tensor_when_keepdim_false_then_output_shape_drops_axis() {
    let shape = vec![3, 4];
    let config = ShapedReductionConfig::new(ReductionOp::Mean, Some(1), false);
    let out_shape = reduction_output_shape(&shape, &config);
    assert_eq!(out_shape, vec![3]);
}

#[test]
fn given_invalid_axis_when_shaped_reduce_then_returns_error() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(5), false);

    let result = shaped_reduce_f32(&input, &shape, &config);
    assert!(result.is_err(), "out-of-bounds axis should fail");
}

#[test]
fn given_shape_mismatch_when_shaped_reduce_then_returns_error() {
    let input = vec![1.0, 2.0, 3.0]; // 3 elements
    let shape = vec![2, 2]; // expects 4
    let config = ShapedReductionConfig::global(ReductionOp::Sum);

    let result = shaped_reduce_f32(&input, &shape, &config);
    assert!(result.is_err(), "shape/data mismatch should fail");
}

#[test]
fn given_large_vector_when_mean_reduced_then_result_is_numerically_stable() {
    let n = 10_000;
    let input: Vec<f32> = (0..n).map(|i| 1e6 + (i as f32) * 0.001).collect();
    let result = reduce_f32(&input, ReductionOp::Mean);
    // f32 precision near 1e6 is ~0.06, so allow wider tolerance
    let expected = 1e6 + (n - 1) as f32 * 0.001 / 2.0;
    assert!(
        (result - expected).abs() < 100.0,
        "mean should be stable; expected ~{expected}, got {result}"
    );
}
