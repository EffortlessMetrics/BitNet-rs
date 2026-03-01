//! Edge-case tests for CPU reduction operations.
//!
//! Tests cover sum, mean, max, min, product, L1/L2 norms
//! with axis reductions and boundary conditions.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::reduction::{ReductionAxis, ReductionKernel};

// ── Global reductions ────────────────────────────────────────────────

#[test]
fn sum_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = ReductionKernel::sum(&data).unwrap();
    assert!((result - 10.0).abs() < 1e-6);
}

#[test]
fn sum_single_element() {
    let result = ReductionKernel::sum(&[42.0]).unwrap();
    assert!((result - 42.0).abs() < 1e-6);
}

#[test]
fn sum_negative_values() {
    let data = vec![-1.0, -2.0, -3.0];
    let result = ReductionKernel::sum(&data).unwrap();
    assert!((result - (-6.0)).abs() < 1e-6);
}

#[test]
fn sum_mixed_values() {
    let data = vec![-5.0, 10.0, -5.0];
    let result = ReductionKernel::sum(&data).unwrap();
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn mean_basic() {
    let data = vec![2.0, 4.0, 6.0, 8.0];
    let result = ReductionKernel::mean(&data).unwrap();
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
fn mean_single() {
    let result = ReductionKernel::mean(&[7.0]).unwrap();
    assert!((result - 7.0).abs() < 1e-6);
}

#[test]
fn max_basic() {
    let data = vec![1.0, 5.0, 3.0, 2.0];
    let result = ReductionKernel::max(&data).unwrap();
    assert!((result.value - 5.0).abs() < 1e-6);
    assert_eq!(result.index, 1);
}

#[test]
fn max_negative() {
    let data = vec![-3.0, -1.0, -5.0];
    let result = ReductionKernel::max(&data).unwrap();
    assert!((result.value - (-1.0)).abs() < 1e-6);
}

#[test]
fn min_basic() {
    let data = vec![3.0, 1.0, 4.0, 1.5];
    let result = ReductionKernel::min(&data).unwrap();
    assert!((result.value - 1.0).abs() < 1e-6);
    assert_eq!(result.index, 1);
}

#[test]
fn product_basic() {
    let data = vec![2.0, 3.0, 4.0];
    let result = ReductionKernel::product(&data).unwrap();
    assert!((result - 24.0).abs() < 1e-6);
}

#[test]
fn product_with_zero() {
    let data = vec![2.0, 0.0, 4.0];
    let result = ReductionKernel::product(&data).unwrap();
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn product_single() {
    let result = ReductionKernel::product(&[5.0]).unwrap();
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
fn l1_norm_basic() {
    let data = vec![-1.0, 2.0, -3.0, 4.0];
    let result = ReductionKernel::l1_norm(&data).unwrap();
    assert!((result - 10.0).abs() < 1e-6);
}

#[test]
fn l2_norm_basic() {
    let data = vec![3.0, 4.0];
    let result = ReductionKernel::l2_norm(&data).unwrap();
    assert!((result - 5.0).abs() < 1e-5);
}

#[test]
fn l2_norm_unit_vector() {
    let data = vec![1.0, 0.0, 0.0];
    let result = ReductionKernel::l2_norm(&data).unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}

// ── Axis reductions ──────────────────────────────────────────────────

#[test]
fn sum_axis_row() {
    // 2x3 matrix: [[1,2,3],[4,5,6]]
    // Row reduction: each row → single value → output length = rows
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = ReductionKernel::sum_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
    // Row 0: 1+2+3=6, Row 1: 4+5+6=15
    assert_eq!(result.len(), 2);
    assert!((result[0] - 6.0).abs() < 1e-6);
    assert!((result[1] - 15.0).abs() < 1e-6);
}

#[test]
fn sum_axis_column() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = ReductionKernel::sum_axis(&data, 2, 3, ReductionAxis::Column).unwrap();
    // Column reduction: each column → single value → output length = cols
    // Col 0: 1+4=5, Col 1: 2+5=7, Col 2: 3+6=9
    assert_eq!(result.len(), 3);
    assert!((result[0] - 5.0).abs() < 1e-6);
    assert!((result[1] - 7.0).abs() < 1e-6);
    assert!((result[2] - 9.0).abs() < 1e-6);
}

#[test]
fn mean_axis_row() {
    let data = vec![2.0, 4.0, 6.0, 8.0]; // 2x2 matrix
    let result = ReductionKernel::mean_axis(&data, 2, 2, ReductionAxis::Row).unwrap();
    // Row reduction: Row 0 mean = (2+4)/2=3, Row 1 mean = (6+8)/2=7
    assert_eq!(result.len(), 2);
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[1] - 7.0).abs() < 1e-6);
}

#[test]
fn max_axis_row() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]; // 2x3
    let result = ReductionKernel::max_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
    // Row reduction: Row 0 max = 5, Row 1 max = 6
    assert_eq!(result.len(), 2);
    assert!((result[0].value - 5.0).abs() < 1e-6);
    assert!((result[1].value - 6.0).abs() < 1e-6);
}

#[test]
fn min_axis_column() {
    let data = vec![3.0, 1.0, 4.0, 2.0]; // 2x2
    let result = ReductionKernel::min_axis(&data, 2, 2, ReductionAxis::Column).unwrap();
    // Column reduction: Col 0 min = min(3,4)=3, Col 1 min = min(1,2)=1
    assert_eq!(result.len(), 2);
    assert!((result[0].value - 3.0).abs() < 1e-6);
    assert!((result[1].value - 1.0).abs() < 1e-6);
}

#[test]
fn l2_norm_axis_row() {
    let data = vec![3.0, 4.0, 0.0, 0.0]; // 2x2
    let result = ReductionKernel::l2_norm_axis(&data, 2, 2, ReductionAxis::Row).unwrap();
    // Row reduction: Row 0 = sqrt(9+16)=5, Row 1 = sqrt(0+0)=0
    assert_eq!(result.len(), 2);
    assert!((result[0] - 5.0).abs() < 1e-5);
}
