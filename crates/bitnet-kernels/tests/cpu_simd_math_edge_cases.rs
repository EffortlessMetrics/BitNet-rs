//! Edge-case tests for CPU SIMD math operations.
//!
//! Tests cover fast approximations (exp, tanh, sigmoid),
//! dot product, vector arithmetic, and L2 norm.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::simd_math::{
    fast_exp_f32, fast_sigmoid_f32, fast_tanh_f32, simd_dot_product, simd_l2_norm, simd_vector_add,
    simd_vector_mul, simd_vector_scale,
};

// ── fast_exp_f32 ─────────────────────────────────────────────────────

#[test]
fn fast_exp_zero() {
    let result = fast_exp_f32(&[0.0]);
    assert!((result[0] - 1.0).abs() < 0.01);
}

#[test]
fn fast_exp_one() {
    let result = fast_exp_f32(&[1.0]);
    assert!((result[0] - std::f32::consts::E).abs() < 0.05);
}

#[test]
fn fast_exp_negative() {
    let result = fast_exp_f32(&[-1.0]);
    let expected = (-1.0f32).exp();
    assert!((result[0] - expected).abs() < 0.05);
}

#[test]
fn fast_exp_batch() {
    let input = vec![0.0, 1.0, -1.0, 2.0];
    let result = fast_exp_f32(&input);
    assert_eq!(result.len(), 4);
    for val in &result {
        assert!(val.is_finite());
        assert!(*val >= 0.0);
    }
}

// ── fast_tanh_f32 ────────────────────────────────────────────────────

#[test]
fn fast_tanh_zero() {
    let result = fast_tanh_f32(&[0.0]);
    assert!(result[0].abs() < 0.01);
}

#[test]
fn fast_tanh_large_positive() {
    let result = fast_tanh_f32(&[10.0]);
    assert!((result[0] - 1.0).abs() < 0.01);
}

#[test]
fn fast_tanh_large_negative() {
    let result = fast_tanh_f32(&[-10.0]);
    assert!((result[0] - (-1.0)).abs() < 0.01);
}

#[test]
fn fast_tanh_bounds() {
    let input = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
    let result = fast_tanh_f32(&input);
    for val in &result {
        assert!(*val >= -1.0 && *val <= 1.0, "tanh out of [-1,1]: {val}");
    }
}

// ── fast_sigmoid_f32 ─────────────────────────────────────────────────

#[test]
fn fast_sigmoid_zero() {
    let result = fast_sigmoid_f32(&[0.0]);
    assert!((result[0] - 0.5).abs() < 0.01);
}

#[test]
fn fast_sigmoid_large_positive() {
    let result = fast_sigmoid_f32(&[10.0]);
    assert!((result[0] - 1.0).abs() < 0.01);
}

#[test]
fn fast_sigmoid_large_negative() {
    let result = fast_sigmoid_f32(&[-10.0]);
    assert!(result[0].abs() < 0.01);
}

#[test]
fn fast_sigmoid_bounds() {
    let input = vec![-100.0, -1.0, 0.0, 1.0, 100.0];
    let result = fast_sigmoid_f32(&input);
    for val in &result {
        assert!(*val >= 0.0 && *val <= 1.0, "sigmoid out of [0,1]: {val}");
    }
}

// ── simd_dot_product ─────────────────────────────────────────────────

#[test]
fn dot_product_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = simd_dot_product(&a, &b);
    assert!((result - 32.0).abs() < 1e-5); // 4+10+18
}

#[test]
fn dot_product_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let result = simd_dot_product(&a, &b);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn dot_product_self() {
    let a = vec![3.0, 4.0];
    let result = simd_dot_product(&a, &a);
    assert!((result - 25.0).abs() < 1e-5);
}

#[test]
fn dot_product_large() {
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.001).collect();
    let result = simd_dot_product(&a, &b);
    assert!(result.is_finite());
}

// ── simd_vector_add ──────────────────────────────────────────────────

#[test]
fn vector_add_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = simd_vector_add(&a, &b);
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn vector_add_negatives() {
    let a = vec![1.0, -2.0, 3.0];
    let b = vec![-1.0, 2.0, -3.0];
    let result = simd_vector_add(&a, &b);
    for val in &result {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// ── simd_vector_mul ──────────────────────────────────────────────────

#[test]
fn vector_mul_basic() {
    let a = vec![2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0];
    let result = simd_vector_mul(&a, &b);
    assert_eq!(result, vec![10.0, 18.0, 28.0]);
}

#[test]
fn vector_mul_zeros() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    let result = simd_vector_mul(&a, &b);
    assert_eq!(result, vec![0.0, 0.0, 0.0]);
}

// ── simd_vector_scale ────────────────────────────────────────────────

#[test]
fn vector_scale_basic() {
    let data = vec![1.0, 2.0, 3.0];
    let result = simd_vector_scale(&data, 2.0);
    assert_eq!(result, vec![2.0, 4.0, 6.0]);
}

#[test]
fn vector_scale_zero() {
    let data = vec![1.0, 2.0, 3.0];
    let result = simd_vector_scale(&data, 0.0);
    assert_eq!(result, vec![0.0, 0.0, 0.0]);
}

#[test]
fn vector_scale_negative() {
    let data = vec![1.0, -2.0, 3.0];
    let result = simd_vector_scale(&data, -1.0);
    assert_eq!(result, vec![-1.0, 2.0, -3.0]);
}

// ── simd_l2_norm ─────────────────────────────────────────────────────

#[test]
fn l2_norm_pythagorean() {
    let data = vec![3.0, 4.0];
    let result = simd_l2_norm(&data);
    assert!((result - 5.0).abs() < 1e-5);
}

#[test]
fn l2_norm_unit() {
    let data = vec![1.0, 0.0, 0.0];
    let result = simd_l2_norm(&data);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn l2_norm_zeros() {
    let data = vec![0.0, 0.0, 0.0];
    let result = simd_l2_norm(&data);
    assert!((result - 0.0).abs() < 1e-6);
}
