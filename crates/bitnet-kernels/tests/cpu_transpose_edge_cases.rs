//! Edge-case tests for CPU transpose and reshape operations.
//!
//! Tests cover 2-D transpose, N-D transpose with permutations,
//! reshape validation, and TransposeConfig.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::transpose::{reshape, transpose_2d, transpose_nd};

// ── 2-D transpose ────────────────────────────────────────────────────

#[test]
fn transpose_2d_square() {
    // [[1,2],[3,4]] → [[1,3],[2,4]]
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_2d(&data, 2, 2);
    assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn transpose_2d_rect() {
    // [[1,2,3],[4,5,6]] (2x3) → [[1,4],[2,5],[3,6]] (3x2)
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_2d(&data, 2, 3);
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_2d_single_row() {
    let data = vec![1.0, 2.0, 3.0]; // 1x3
    let result = transpose_2d(&data, 1, 3);
    assert_eq!(result, vec![1.0, 2.0, 3.0]); // 3x1
}

#[test]
fn transpose_2d_single_col() {
    let data = vec![1.0, 2.0, 3.0]; // 3x1
    let result = transpose_2d(&data, 3, 1);
    assert_eq!(result, vec![1.0, 2.0, 3.0]); // 1x3
}

#[test]
fn transpose_2d_single_element() {
    let data = vec![42.0]; // 1x1
    let result = transpose_2d(&data, 1, 1);
    assert_eq!(result, vec![42.0]);
}

#[test]
fn transpose_2d_identity_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let transposed = transpose_2d(&data, 2, 3); // 3x2
    let back = transpose_2d(&transposed, 3, 2); // 2x3
    assert_eq!(back, data);
}

// ── N-D transpose ────────────────────────────────────────────────────

#[test]
fn transpose_nd_2d_equiv() {
    let data = vec![1.0, 2.0, 3.0, 4.0]; // shape [2,2]
    let result = transpose_nd(&data, &[2, 2], &[1, 0]);
    assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn transpose_nd_3d_swap_last_two() {
    // shape [2,2,3] → perm [0,2,1] → [2,3,2]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let result = transpose_nd(&data, &[2, 2, 3], &[0, 2, 1]);
    assert_eq!(result.len(), 12);
    // First block [2,3] → [3,2]: [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 4.0);
    assert_eq!(result[2], 2.0);
    assert_eq!(result[3], 5.0);
}

#[test]
fn transpose_nd_identity_perm() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_nd(&data, &[2, 3], &[0, 1]); // identity
    assert_eq!(result, data);
}

// ── Reshape ──────────────────────────────────────────────────────────

#[test]
fn reshape_same_total() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let result = reshape(&data, &[2, 3], &[3, 2]).unwrap();
    assert_eq!(result, data); // Same flat data, just reinterpreted
}

#[test]
fn reshape_flatten() {
    let data = vec![1.0, 2.0, 3.0, 4.0]; // [2,2]
    let result = reshape(&data, &[2, 2], &[4]).unwrap();
    assert_eq!(result, data);
}

#[test]
fn reshape_expand_dims() {
    let data = vec![1.0, 2.0, 3.0]; // [3]
    let result = reshape(&data, &[3], &[1, 3]).unwrap();
    assert_eq!(result, data);
}

#[test]
fn reshape_mismatched_total_fails() {
    let data = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements
    let result = reshape(&data, &[2, 2], &[3, 2]); // needs 6
    assert!(result.is_err());
}

#[test]
fn reshape_single_element() {
    let data = vec![42.0];
    let result = reshape(&data, &[1], &[1, 1, 1]).unwrap();
    assert_eq!(result, vec![42.0]);
}
