//! Edge-case tests for CPU scatter/gather operations.
//!
//! Tests cover 1-D, 2-D, and N-D scatter/gather variants
//! plus convenience wrappers (scatter_add, scatter_max, index_select).

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::scatter_gather::{
    ScatterReduce, cpu_gather, cpu_scatter, gather_1d, gather_2d, index_select, scatter_1d,
    scatter_2d, scatter_add, scatter_max,
};

// ── 1-D operations ───────────────────────────────────────────────────

#[test]
fn gather_1d_basic() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices = vec![0, 2, 4];
    let result = gather_1d(&data, &indices).unwrap();
    assert_eq!(result, vec![10.0, 30.0, 50.0]);
}

#[test]
fn gather_1d_single() {
    let data = vec![42.0];
    let result = gather_1d(&data, &[0]).unwrap();
    assert_eq!(result, vec![42.0]);
}

#[test]
fn gather_1d_repeated_indices() {
    let data = vec![1.0, 2.0, 3.0];
    let result = gather_1d(&data, &[1, 1, 1]).unwrap();
    assert_eq!(result, vec![2.0, 2.0, 2.0]);
}

#[test]
fn scatter_1d_basic() {
    let mut data = vec![0.0; 5];
    scatter_1d(&mut data, &[1, 3], &[10.0, 20.0]).unwrap();
    assert_eq!(data, vec![0.0, 10.0, 0.0, 20.0, 0.0]);
}

#[test]
fn scatter_add_basic() {
    let mut data = vec![1.0, 2.0, 3.0];
    scatter_add(&mut data, &[0, 2], &[10.0, 20.0]).unwrap();
    assert_eq!(data, vec![11.0, 2.0, 23.0]);
}

#[test]
fn scatter_add_same_index() {
    let mut data = vec![0.0; 3];
    scatter_add(&mut data, &[1, 1, 1], &[1.0, 2.0, 3.0]).unwrap();
    assert!((data[1] - 6.0).abs() < 1e-6);
}

#[test]
fn scatter_max_basic() {
    let mut data = vec![5.0, 5.0, 5.0];
    scatter_max(&mut data, &[0, 1, 2], &[3.0, 8.0, 2.0]).unwrap();
    assert_eq!(data, vec![5.0, 8.0, 5.0]);
}

#[test]
fn scatter_max_all_smaller() {
    let mut data = vec![10.0, 10.0];
    scatter_max(&mut data, &[0, 1], &[5.0, 5.0]).unwrap();
    assert_eq!(data, vec![10.0, 10.0]);
}

// ── 2-D Vec-of-Vec operations ────────────────────────────────────────

#[test]
fn gather_2d_basic() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let result = gather_2d(&data, &[0, 2]).unwrap();
    assert_eq!(result, vec![vec![1.0, 2.0], vec![5.0, 6.0]]);
}

#[test]
fn scatter_2d_basic() {
    let mut data = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
    scatter_2d(&mut data, &[1], &[vec![10.0, 20.0]]).unwrap();
    assert_eq!(data[1], vec![10.0, 20.0]);
    assert_eq!(data[0], vec![0.0, 0.0]);
}

// ── 2-D flat array operations ────────────────────────────────────────

#[test]
fn cpu_gather_axis0() {
    // 3x2 source: [[1,2],[3,4],[5,6]]
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Gather rows 0 and 2 → idx_shape [2, 2] (2 rows, cols must match src cols=2)
    let indices = vec![0, 0, 2, 2]; // row indices repeated per col
    let mut output = vec![0.0; 4]; // 2x2 result
    cpu_gather(&src, [3, 2], &indices, [2, 2], 0, true, &mut output).unwrap();
    assert_eq!(output, vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn cpu_scatter_assign_axis0() {
    let src = vec![10.0, 20.0, 30.0, 40.0]; // 2x2
    let indices = vec![0, 0, 2, 2]; // idx_shape [2, 2]
    let mut dst = vec![0.0; 6]; // 3x2
    cpu_scatter(&src, &indices, [2, 2], &mut dst, [3, 2], 0, ScatterReduce::Assign, true).unwrap();
    assert_eq!(dst[0], 10.0);
    assert_eq!(dst[1], 20.0);
    assert_eq!(dst[4], 30.0);
    assert_eq!(dst[5], 40.0);
}

// ── index_select ─────────────────────────────────────────────────────

#[test]
fn index_select_basic() {
    // data has 6 elements, dim_size=3 means 3 groups of 2 elements
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = index_select(&data, 3, &[0, 2]).unwrap();
    // Group 0: [1,2], Group 2: [5,6]
    assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn index_select_single_row() {
    let data = vec![10.0, 20.0, 30.0, 40.0]; // dim_size=2 → 2 groups of 2
    let result = index_select(&data, 2, &[1]).unwrap();
    assert_eq!(result, vec![30.0, 40.0]);
}

#[test]
fn index_select_all_rows() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // dim_size=3 → 3 groups of 2
    let result = index_select(&data, 3, &[0, 1, 2]).unwrap();
    assert_eq!(result, data);
}
