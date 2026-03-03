#![allow(clippy::cast_precision_loss)]
//! Comprehensive tests for bitnet-cuda-elementwise.
//!
//! Coverage targets:
//! - Arithmetic correctness (add, sub, mul, div, fma)
//! - Broadcasting (scalar-tensor, vector-tensor, tensor-tensor)
//! - In-place variants
//! - Activation functions (relu, gelu, silu, sigmoid, tanh)
//! - Edge cases (empty, single-element, large)
//! - NaN / Inf propagation
//! - Error paths

use bitnet_cuda_elementwise::{
    Activation, BroadcastShape, ElementWiseError, add, add_inplace, apply_activation,
    apply_activation_inplace, div, div_inplace, fma, fma_inplace, mul, mul_inplace, sub,
    sub_inplace,
};

// ════════════════════════════════════════════════════════════════════
// Helper
// ════════════════════════════════════════════════════════════════════

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        assert!((x - y).abs() <= tol, "index {i}: {x} vs {y} (diff {})", (x - y).abs());
    }
}

// ════════════════════════════════════════════════════════════════════
// 1. Broadcast shape resolution
// ════════════════════════════════════════════════════════════════════

#[test]
fn broadcast_same_length() {
    assert_eq!(BroadcastShape::resolve(4, 4).unwrap(), BroadcastShape::Same(4));
}

#[test]
fn broadcast_scalar_left() {
    assert_eq!(BroadcastShape::resolve(1, 5).unwrap(), BroadcastShape::ScalarLeft(5));
}

#[test]
fn broadcast_scalar_right() {
    assert_eq!(BroadcastShape::resolve(5, 1).unwrap(), BroadcastShape::ScalarRight(5));
}

#[test]
fn broadcast_vector_right() {
    let s = BroadcastShape::resolve(8, 4).unwrap();
    assert_eq!(s, BroadcastShape::VectorRight { total_len: 8, rhs_len: 4 });
    assert_eq!(s.output_len(), 8);
}

#[test]
fn broadcast_vector_left() {
    let s = BroadcastShape::resolve(3, 9).unwrap();
    assert_eq!(s, BroadcastShape::VectorLeft { total_len: 9, lhs_len: 3 });
}

#[test]
fn broadcast_incompatible() {
    assert!(BroadcastShape::resolve(3, 5).is_err());
}

#[test]
fn broadcast_empty_lhs() {
    assert!(matches!(BroadcastShape::resolve(0, 5), Err(ElementWiseError::EmptyTensor)));
}

#[test]
fn broadcast_empty_rhs() {
    assert!(matches!(BroadcastShape::resolve(5, 0), Err(ElementWiseError::EmptyTensor)));
}

#[test]
fn broadcast_both_empty() {
    assert!(matches!(BroadcastShape::resolve(0, 0), Err(ElementWiseError::EmptyTensor)));
}

#[test]
fn broadcast_both_scalar() {
    assert_eq!(BroadcastShape::resolve(1, 1).unwrap(), BroadcastShape::Same(1));
}

// ════════════════════════════════════════════════════════════════════
// 2. Addition
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_same_length() {
    let r = add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
    approx_eq(&r, &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn add_scalar_right() {
    let r = add(&[1.0, 2.0, 3.0], &[10.0]).unwrap();
    approx_eq(&r, &[11.0, 12.0, 13.0], 1e-6);
}

#[test]
fn add_scalar_left() {
    let r = add(&[10.0], &[1.0, 2.0, 3.0]).unwrap();
    approx_eq(&r, &[11.0, 12.0, 13.0], 1e-6);
}

#[test]
fn add_vector_broadcast() {
    let r = add(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0]).unwrap();
    approx_eq(&r, &[11.0, 22.0, 13.0, 24.0], 1e-6);
}

#[test]
fn add_inplace_scalar() {
    let mut v = vec![1.0, 2.0, 3.0];
    add_inplace(&mut v, &[5.0]).unwrap();
    approx_eq(&v, &[6.0, 7.0, 8.0], 1e-6);
}

#[test]
fn add_inplace_same() {
    let mut v = vec![1.0, 2.0];
    add_inplace(&mut v, &[3.0, 4.0]).unwrap();
    approx_eq(&v, &[4.0, 6.0], 1e-6);
}

#[test]
fn add_inplace_rejects_larger_rhs() {
    let mut v = vec![1.0];
    assert!(add_inplace(&mut v, &[1.0, 2.0, 3.0]).is_err());
}

#[test]
fn add_empty_error() {
    assert!(add(&[], &[1.0]).is_err());
}

// ════════════════════════════════════════════════════════════════════
// 3. Subtraction
// ════════════════════════════════════════════════════════════════════

#[test]
fn sub_same_length() {
    let r = sub(&[10.0, 20.0], &[3.0, 7.0]).unwrap();
    approx_eq(&r, &[7.0, 13.0], 1e-6);
}

#[test]
fn sub_scalar_right() {
    let r = sub(&[5.0, 10.0], &[2.0]).unwrap();
    approx_eq(&r, &[3.0, 8.0], 1e-6);
}

#[test]
fn sub_inplace_basic() {
    let mut v = vec![10.0, 20.0, 30.0];
    sub_inplace(&mut v, &[1.0, 2.0, 3.0]).unwrap();
    approx_eq(&v, &[9.0, 18.0, 27.0], 1e-6);
}

#[test]
fn sub_incompatible_shapes() {
    assert!(sub(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
}

// ════════════════════════════════════════════════════════════════════
// 4. Multiplication
// ════════════════════════════════════════════════════════════════════

#[test]
fn mul_same_length() {
    let r = mul(&[2.0, 3.0], &[4.0, 5.0]).unwrap();
    approx_eq(&r, &[8.0, 15.0], 1e-6);
}

#[test]
fn mul_scalar_broadcast() {
    let r = mul(&[1.0, 2.0, 3.0], &[3.0]).unwrap();
    approx_eq(&r, &[3.0, 6.0, 9.0], 1e-6);
}

#[test]
fn mul_vector_broadcast() {
    let r = mul(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[10.0, 100.0, 1000.0]).unwrap();
    approx_eq(&r, &[10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0], 1e-6);
}

#[test]
fn mul_inplace_basic() {
    let mut v = vec![2.0, 3.0];
    mul_inplace(&mut v, &[4.0, 5.0]).unwrap();
    approx_eq(&v, &[8.0, 15.0], 1e-6);
}

#[test]
fn mul_by_zero() {
    let r = mul(&[1.0, 2.0], &[0.0]).unwrap();
    approx_eq(&r, &[0.0, 0.0], 1e-6);
}

// ════════════════════════════════════════════════════════════════════
// 5. Division
// ════════════════════════════════════════════════════════════════════

#[test]
fn div_same_length() {
    let r = div(&[10.0, 20.0], &[2.0, 5.0]).unwrap();
    approx_eq(&r, &[5.0, 4.0], 1e-6);
}

#[test]
fn div_scalar_right() {
    let r = div(&[10.0, 20.0, 30.0], &[10.0]).unwrap();
    approx_eq(&r, &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn div_by_zero_error() {
    assert!(matches!(div(&[1.0, 2.0], &[0.0]), Err(ElementWiseError::DivisionByZero)));
}

#[test]
fn div_zero_by_zero_is_nan() {
    // IEEE-754: 0.0 / 0.0 = NaN — no error
    let r = div(&[0.0], &[0.0]).unwrap();
    assert!(r[0].is_nan());
}

#[test]
fn div_inplace_basic() {
    let mut v = vec![10.0, 20.0];
    div_inplace(&mut v, &[2.0, 4.0]).unwrap();
    approx_eq(&v, &[5.0, 5.0], 1e-6);
}

#[test]
fn div_inplace_by_zero() {
    let mut v = vec![1.0];
    assert!(div_inplace(&mut v, &[0.0]).is_err());
}

// ════════════════════════════════════════════════════════════════════
// 6. Fused multiply-add
// ════════════════════════════════════════════════════════════════════

#[test]
fn fma_basic() {
    // a*b + c = 2*3 + 1 = 7
    let r = fma(&[2.0], &[3.0], &[1.0]).unwrap();
    approx_eq(&r, &[7.0], 1e-6);
}

#[test]
fn fma_vector() {
    let r = fma(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]).unwrap();
    // 1*4+7=11, 2*5+8=18, 3*6+9=27
    approx_eq(&r, &[11.0, 18.0, 27.0], 1e-6);
}

#[test]
fn fma_broadcast_c_scalar() {
    let r = fma(&[1.0, 2.0], &[3.0, 4.0], &[10.0]).unwrap();
    approx_eq(&r, &[13.0, 18.0], 1e-6);
}

#[test]
fn fma_broadcast_b_scalar() {
    let r = fma(&[1.0, 2.0, 3.0], &[2.0], &[0.0, 0.0, 0.0]).unwrap();
    approx_eq(&r, &[2.0, 4.0, 6.0], 1e-6);
}

#[test]
fn fma_all_scalars() {
    let r = fma(&[3.0], &[4.0], &[5.0]).unwrap();
    approx_eq(&r, &[17.0], 1e-6);
}

#[test]
fn fma_length_mismatch() {
    assert!(matches!(
        fma(&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0]),
        Err(ElementWiseError::FmaLengthMismatch { .. })
    ));
}

#[test]
fn fma_inplace_basic() {
    let mut a = vec![2.0, 3.0];
    fma_inplace(&mut a, &[4.0, 5.0], &[1.0, 1.0]).unwrap();
    approx_eq(&a, &[9.0, 16.0], 1e-6);
}

#[test]
fn fma_inplace_scalar_broadcast() {
    let mut a = vec![1.0, 2.0, 3.0];
    fma_inplace(&mut a, &[2.0], &[10.0]).unwrap();
    approx_eq(&a, &[12.0, 14.0, 16.0], 1e-6);
}

#[test]
fn fma_inplace_rejects_larger_b() {
    let mut a = vec![1.0];
    assert!(fma_inplace(&mut a, &[1.0, 2.0, 3.0], &[1.0]).is_err());
}

// ════════════════════════════════════════════════════════════════════
// 7. Activation functions — basic correctness
// ════════════════════════════════════════════════════════════════════

#[test]
fn relu_positive() {
    let r = apply_activation(Activation::ReLU, &[1.0, 2.0, 3.0]);
    approx_eq(&r, &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn relu_negative() {
    let r = apply_activation(Activation::ReLU, &[-1.0, -0.5, 0.0]);
    approx_eq(&r, &[0.0, 0.0, 0.0], 1e-6);
}

#[test]
fn relu_mixed() {
    let r = apply_activation(Activation::ReLU, &[-2.0, 0.0, 3.0]);
    approx_eq(&r, &[0.0, 0.0, 3.0], 1e-6);
}

#[test]
fn sigmoid_zero() {
    let r = apply_activation(Activation::Sigmoid, &[0.0]);
    approx_eq(&r, &[0.5], 1e-6);
}

#[test]
fn sigmoid_large_positive() {
    let r = apply_activation(Activation::Sigmoid, &[100.0]);
    assert!((r[0] - 1.0).abs() < 1e-6);
}

#[test]
fn sigmoid_large_negative() {
    let r = apply_activation(Activation::Sigmoid, &[-100.0]);
    assert!(r[0].abs() < 1e-6);
}

#[test]
fn tanh_zero() {
    let r = apply_activation(Activation::Tanh, &[0.0]);
    approx_eq(&r, &[0.0], 1e-6);
}

#[test]
fn tanh_symmetry() {
    let pos = apply_activation(Activation::Tanh, &[1.5]);
    let neg = apply_activation(Activation::Tanh, &[-1.5]);
    approx_eq(&[pos[0] + neg[0]], &[0.0], 1e-6);
}

#[test]
fn gelu_zero() {
    let r = apply_activation(Activation::GELU, &[0.0]);
    approx_eq(&r, &[0.0], 1e-5);
}

#[test]
fn gelu_positive() {
    let r = apply_activation(Activation::GELU, &[1.0]);
    // GELU(1.0) ≈ 0.8413
    assert!(r[0] > 0.84 && r[0] < 0.85);
}

#[test]
fn gelu_negative() {
    let r = apply_activation(Activation::GELU, &[-1.0]);
    // GELU(-1.0) ≈ -0.1587
    assert!(r[0] > -0.17 && r[0] < -0.15);
}

#[test]
fn silu_zero() {
    let r = apply_activation(Activation::SiLU, &[0.0]);
    approx_eq(&r, &[0.0], 1e-6);
}

#[test]
fn silu_positive() {
    let r = apply_activation(Activation::SiLU, &[2.0]);
    // SiLU(2) = 2 * sigmoid(2) ≈ 2 * 0.8808 ≈ 1.7616
    assert!(r[0] > 1.76 && r[0] < 1.77);
}

#[test]
fn silu_negative() {
    let r = apply_activation(Activation::SiLU, &[-2.0]);
    // SiLU(-2) = -2 * sigmoid(-2) ≈ -2 * 0.1192 ≈ -0.2384
    assert!(r[0] > -0.24 && r[0] < -0.23);
}

// ════════════════════════════════════════════════════════════════════
// 8. In-place activations
// ════════════════════════════════════════════════════════════════════

#[test]
fn relu_inplace() {
    let mut v = vec![-1.0, 0.0, 1.0];
    apply_activation_inplace(Activation::ReLU, &mut v);
    approx_eq(&v, &[0.0, 0.0, 1.0], 1e-6);
}

#[test]
fn sigmoid_inplace() {
    let mut v = vec![0.0];
    apply_activation_inplace(Activation::Sigmoid, &mut v);
    approx_eq(&v, &[0.5], 1e-6);
}

#[test]
fn tanh_inplace() {
    let mut v = vec![0.0, 1.0];
    let expected = apply_activation(Activation::Tanh, &[0.0, 1.0]);
    apply_activation_inplace(Activation::Tanh, &mut v);
    approx_eq(&v, &expected, 1e-6);
}

#[test]
fn gelu_inplace() {
    let mut v = vec![-1.0, 0.0, 1.0];
    let expected = apply_activation(Activation::GELU, &[-1.0, 0.0, 1.0]);
    apply_activation_inplace(Activation::GELU, &mut v);
    approx_eq(&v, &expected, 1e-6);
}

#[test]
fn silu_inplace() {
    let mut v = vec![-1.0, 0.0, 1.0];
    let expected = apply_activation(Activation::SiLU, &[-1.0, 0.0, 1.0]);
    apply_activation_inplace(Activation::SiLU, &mut v);
    approx_eq(&v, &expected, 1e-6);
}

// ════════════════════════════════════════════════════════════════════
// 9. NaN propagation
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_nan_propagates() {
    let r = add(&[f32::NAN, 1.0], &[2.0, f32::NAN]).unwrap();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
}

#[test]
fn sub_nan_propagates() {
    let r = sub(&[f32::NAN], &[1.0]).unwrap();
    assert!(r[0].is_nan());
}

#[test]
fn mul_nan_propagates() {
    let r = mul(&[f32::NAN, 2.0], &[3.0, f32::NAN]).unwrap();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
}

#[test]
fn div_nan_numerator() {
    let r = div(&[f32::NAN], &[1.0]).unwrap();
    assert!(r[0].is_nan());
}

#[test]
fn div_nan_denominator() {
    let r = div(&[1.0], &[f32::NAN]).unwrap();
    assert!(r[0].is_nan());
}

#[test]
fn fma_nan_propagates() {
    let r = fma(&[f32::NAN], &[2.0], &[3.0]).unwrap();
    assert!(r[0].is_nan());
}

#[test]
fn relu_nan() {
    let r = apply_activation(Activation::ReLU, &[f32::NAN]);
    assert!(r[0].is_nan());
}

#[test]
fn sigmoid_nan() {
    let r = apply_activation(Activation::Sigmoid, &[f32::NAN]);
    assert!(r[0].is_nan());
}

#[test]
fn gelu_nan() {
    let r = apply_activation(Activation::GELU, &[f32::NAN]);
    assert!(r[0].is_nan());
}

#[test]
fn tanh_nan() {
    let r = apply_activation(Activation::Tanh, &[f32::NAN]);
    assert!(r[0].is_nan());
}

#[test]
fn silu_nan() {
    let r = apply_activation(Activation::SiLU, &[f32::NAN]);
    assert!(r[0].is_nan());
}

// ════════════════════════════════════════════════════════════════════
// 10. Infinity handling
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_infinity() {
    let r = add(&[f32::INFINITY], &[1.0]).unwrap();
    assert!(r[0] == f32::INFINITY);
}

#[test]
fn sub_infinity_cancel() {
    let r = sub(&[f32::INFINITY], &[f32::INFINITY]).unwrap();
    assert!(r[0].is_nan()); // inf - inf = NaN
}

#[test]
fn mul_infinity_by_zero() {
    let r = mul(&[f32::INFINITY], &[0.0]).unwrap();
    assert!(r[0].is_nan()); // inf * 0 = NaN
}

#[test]
fn div_by_infinity() {
    let r = div(&[1.0], &[f32::INFINITY]).unwrap();
    approx_eq(&r, &[0.0], 1e-6);
}

#[test]
fn relu_neg_infinity() {
    let r = apply_activation(Activation::ReLU, &[f32::NEG_INFINITY]);
    approx_eq(&r, &[0.0], 1e-6);
}

#[test]
fn sigmoid_infinity() {
    let r = apply_activation(Activation::Sigmoid, &[f32::INFINITY]);
    approx_eq(&r, &[1.0], 1e-6);
}

// ════════════════════════════════════════════════════════════════════
// 11. Single element tensors
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_single_elements() {
    let r = add(&[3.0], &[4.0]).unwrap();
    approx_eq(&r, &[7.0], 1e-6);
}

#[test]
fn sub_single_elements() {
    let r = sub(&[10.0], &[3.0]).unwrap();
    approx_eq(&r, &[7.0], 1e-6);
}

#[test]
fn mul_single_elements() {
    let r = mul(&[3.0], &[4.0]).unwrap();
    approx_eq(&r, &[12.0], 1e-6);
}

#[test]
fn div_single_elements() {
    let r = div(&[12.0], &[4.0]).unwrap();
    approx_eq(&r, &[3.0], 1e-6);
}

// ════════════════════════════════════════════════════════════════════
// 12. Activation: all_variants coverage
// ════════════════════════════════════════════════════════════════════

#[test]
fn all_variants_returns_five() {
    assert_eq!(Activation::all_variants().len(), 5);
}

#[test]
fn all_activations_on_zero() {
    for &act in Activation::all_variants() {
        let r = act.apply_scalar(0.0);
        assert!(r.is_finite(), "activation {act:?} on 0.0 is not finite");
    }
}

#[test]
fn all_activations_on_positive() {
    for &act in Activation::all_variants() {
        let r = act.apply_scalar(1.0);
        assert!(r.is_finite(), "activation {act:?} on 1.0 is not finite");
        assert!(r > 0.0, "activation {act:?}(1.0) should be positive, got {r}");
    }
}

#[test]
fn all_activations_inplace_matches_outofplace() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    for &act in Activation::all_variants() {
        let expected = apply_activation(act, &input);
        let mut buf = input.clone();
        apply_activation_inplace(act, &mut buf);
        approx_eq(&buf, &expected, 1e-6);
    }
}

// ════════════════════════════════════════════════════════════════════
// 13. Error display
// ════════════════════════════════════════════════════════════════════

#[test]
fn error_display_shape_mismatch() {
    let e = ElementWiseError::ShapeMismatch { lhs: vec![3], rhs: vec![5] };
    assert!(e.to_string().contains("shape mismatch"));
}

#[test]
fn error_display_empty_tensor() {
    let e = ElementWiseError::EmptyTensor;
    assert!(e.to_string().contains("empty tensor"));
}

#[test]
fn error_display_div_zero() {
    let e = ElementWiseError::DivisionByZero;
    assert!(e.to_string().contains("division by zero"));
}

#[test]
fn error_display_fma_mismatch() {
    let e = ElementWiseError::FmaLengthMismatch { a_len: 1, b_len: 2, c_len: 3 };
    let s = e.to_string();
    assert!(s.contains("FMA") && s.contains('1') && s.contains('2') && s.contains('3'));
}

// ════════════════════════════════════════════════════════════════════
// 14. Inplace vector broadcast
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_inplace_vector_broadcast() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0];
    add_inplace(&mut v, &[10.0, 20.0]).unwrap();
    approx_eq(&v, &[11.0, 22.0, 13.0, 24.0], 1e-6);
}

#[test]
fn mul_inplace_vector_broadcast() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    mul_inplace(&mut v, &[10.0, 100.0, 1000.0]).unwrap();
    approx_eq(&v, &[10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0], 1e-6);
}

#[test]
fn sub_inplace_vector_broadcast() {
    let mut v = vec![10.0, 20.0, 30.0, 40.0];
    sub_inplace(&mut v, &[1.0, 2.0]).unwrap();
    approx_eq(&v, &[9.0, 18.0, 29.0, 38.0], 1e-6);
}

#[test]
fn div_inplace_vector_broadcast() {
    let mut v = vec![10.0, 20.0, 30.0, 40.0];
    div_inplace(&mut v, &[2.0, 5.0]).unwrap();
    approx_eq(&v, &[5.0, 4.0, 15.0, 8.0], 1e-6);
}

// ════════════════════════════════════════════════════════════════════
// 15. Large tensor smoke test
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_large_tensor() {
    let n = 10_000_usize;
    let a: Vec<f32> = (0..n).map(|i| (i & 0xFF_FFFF) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| ((n - i) & 0xFF_FFFF) as f32).collect();
    let r = add(&a, &b).unwrap();
    assert_eq!(r.len(), n);
    let expected = (n & 0xFF_FFFF) as f32;
    for &v in &r {
        approx_eq(&[v], &[expected], 1e-6);
    }
}

#[test]
fn activation_large_tensor() {
    let n = 10_000_usize;
    let data: Vec<f32> = (0..n).map(|i| ((i & 0xFF_FFFF) as f32).mul_add(0.001, -5.0)).collect();
    let r = apply_activation(Activation::ReLU, &data);
    assert_eq!(r.len(), n);
    for &v in &r {
        assert!(v >= 0.0);
    }
}

// ════════════════════════════════════════════════════════════════════
// 16. Negative tests — shape mismatches across ops
// ════════════════════════════════════════════════════════════════════

#[test]
fn sub_empty() {
    assert!(sub(&[], &[]).is_err());
}

#[test]
fn mul_empty() {
    assert!(mul(&[], &[1.0]).is_err());
}

#[test]
fn div_empty() {
    assert!(div(&[1.0], &[]).is_err());
}

#[test]
fn fma_empty_a() {
    assert!(fma(&[], &[1.0], &[1.0]).is_err());
}

// ════════════════════════════════════════════════════════════════════
// 17. Identity properties
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_zero_identity() {
    let a = vec![1.0, -2.0, 3.5];
    let r = add(&a, &[0.0]).unwrap();
    approx_eq(&r, &a, 1e-6);
}

#[test]
fn mul_one_identity() {
    let a = vec![1.0, -2.0, 3.5];
    let r = mul(&a, &[1.0]).unwrap();
    approx_eq(&r, &a, 1e-6);
}

#[test]
fn sub_self_is_zero() {
    let a = vec![1.0, -2.0, 3.5];
    let r = sub(&a, &a).unwrap();
    approx_eq(&r, &[0.0, 0.0, 0.0], 1e-6);
}

#[test]
fn div_self_is_one() {
    let a = vec![1.0, -2.0, 3.5];
    let r = div(&a, &a).unwrap();
    approx_eq(&r, &[1.0, 1.0, 1.0], 1e-6);
}

// ════════════════════════════════════════════════════════════════════
// 18. Commutativity / associativity
// ════════════════════════════════════════════════════════════════════

#[test]
fn add_commutative() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let ab = add(&a, &b).unwrap();
    let ba = add(&b, &a).unwrap();
    approx_eq(&ab, &ba, 1e-6);
}

#[test]
fn mul_commutative() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let ab = mul(&a, &b).unwrap();
    let ba = mul(&b, &a).unwrap();
    approx_eq(&ab, &ba, 1e-6);
}

// ════════════════════════════════════════════════════════════════════
// 19. Property tests
// ════════════════════════════════════════════════════════════════════

mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn finite_f32() -> impl Strategy<Value = f32> {
        -1e6_f32..1e6_f32
    }

    fn small_finite_f32() -> impl Strategy<Value = f32> {
        -1e3_f32..1e3_f32
    }

    fn finite_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(finite_f32(), 1..=max_len)
    }

    fn small_finite_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(small_finite_f32(), 1..=max_len)
    }

    proptest! {
        #[test]
        fn prop_add_commutative(a in finite_vec(64), b in finite_vec(64)) {
            if a.len() == b.len() {
                let ab = add(&a, &b).unwrap();
                let ba = add(&b, &a).unwrap();
                approx_eq(&ab, &ba, 1e-4);
            }
        }

        #[test]
        fn prop_mul_commutative(a in small_finite_vec(64), b in small_finite_vec(64)) {
            if a.len() == b.len() {
                let ab = mul(&a, &b).unwrap();
                let ba = mul(&b, &a).unwrap();
                approx_eq(&ab, &ba, 1e-4);
            }
        }

        #[test]
        fn prop_add_zero_identity(a in finite_vec(64)) {
            let r = add(&a, &[0.0]).unwrap();
            approx_eq(&r, &a, 1e-6);
        }

        #[test]
        fn prop_mul_one_identity(a in finite_vec(64)) {
            let r = mul(&a, &[1.0]).unwrap();
            approx_eq(&r, &a, 1e-4);
        }

        #[test]
        fn prop_sub_self_zero(a in finite_vec(64)) {
            let r = sub(&a, &a).unwrap();
            let zeros = vec![0.0; a.len()];
            approx_eq(&r, &zeros, 1e-4);
        }

        #[test]
        fn prop_relu_non_negative(v in finite_vec(128)) {
            let r = apply_activation(Activation::ReLU, &v);
            for &x in &r {
                prop_assert!(x >= 0.0, "ReLU output {x} is negative");
            }
        }

        #[test]
        fn prop_sigmoid_bounded(v in finite_vec(128)) {
            let r = apply_activation(Activation::Sigmoid, &v);
            for &x in &r {
                prop_assert!((0.0..=1.0).contains(&x), "sigmoid {x} out of [0,1]");
            }
        }

        #[test]
        fn prop_tanh_bounded(v in finite_vec(128)) {
            let r = apply_activation(Activation::Tanh, &v);
            for &x in &r {
                prop_assert!((-1.0..=1.0).contains(&x), "tanh {x} out of [-1,1]");
            }
        }

        #[test]
        fn prop_inplace_matches_outofplace_add(
            a in finite_vec(64),
            b_elem in finite_f32()
        ) {
            let expected = add(&a, &[b_elem]).unwrap();
            let mut buf = a;
            add_inplace(&mut buf, &[b_elem]).unwrap();
            approx_eq(&buf, &expected, 1e-4);
        }

        #[test]
        fn prop_fma_scalar_identity(
            a in finite_vec(32),
            c in finite_vec(32)
        ) {
            // a * 1.0 + c  == add(a, c) when same length
            if a.len() == c.len() {
                let fma_r = fma(&a, &[1.0], &c).unwrap();
                let add_r = add(&a, &c).unwrap();
                approx_eq(&fma_r, &add_r, 1e-3);
            }
        }

        #[test]
        fn prop_broadcast_output_len(
            lhs_len in 1_usize..=256,
            rhs_len in 1_usize..=256
        ) {
            if let Ok(shape) = BroadcastShape::resolve(lhs_len, rhs_len) {
                let out = shape.output_len();
                prop_assert!(out >= lhs_len.max(rhs_len));
            }
        }
    }
}
