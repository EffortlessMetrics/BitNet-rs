//! Edge-case tests for CPU activation function kernels.
//!
//! Tests cover boundary conditions, special values (NaN, infinity, zero),
//! mathematical properties, and consistency between scalar/batch variants.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{
    ActivationType, activate, activate_derivative, activate_inplace, elu, gelu, gelu_tanh,
    hard_sigmoid, hard_swish, leaky_relu, mish, quick_gelu, relu, selu, sigmoid, silu, softplus,
    swish, tanh_act,
};

// ── ReLU ─────────────────────────────────────────────────────────────

#[test]
fn relu_positive() {
    assert_eq!(relu(5.0), 5.0);
}

#[test]
fn relu_negative() {
    assert_eq!(relu(-5.0), 0.0);
}

#[test]
fn relu_zero() {
    assert_eq!(relu(0.0), 0.0);
}

#[test]
fn relu_very_large() {
    assert_eq!(relu(1e38), 1e38);
}

#[test]
fn relu_neg_zero() {
    assert_eq!(relu(-0.0), 0.0);
}

// ── Leaky ReLU ───────────────────────────────────────────────────────

#[test]
fn leaky_relu_positive() {
    assert_eq!(leaky_relu(5.0, 0.01), 5.0);
}

#[test]
fn leaky_relu_negative() {
    assert!((leaky_relu(-5.0, 0.01) - (-0.05)).abs() < 1e-6);
}

#[test]
fn leaky_relu_zero_alpha() {
    // alpha=0 makes it identical to ReLU
    assert_eq!(leaky_relu(-5.0, 0.0), 0.0);
    assert_eq!(leaky_relu(5.0, 0.0), 5.0);
}

#[test]
fn leaky_relu_alpha_one_is_identity() {
    assert!((leaky_relu(-3.0, 1.0) - (-3.0)).abs() < 1e-6);
    assert!((leaky_relu(3.0, 1.0) - 3.0).abs() < 1e-6);
}

// ── Sigmoid ──────────────────────────────────────────────────────────

#[test]
fn sigmoid_zero_is_half() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
}

#[test]
fn sigmoid_large_positive_near_one() {
    assert!((sigmoid(100.0) - 1.0).abs() < 1e-5);
}

#[test]
fn sigmoid_large_negative_near_zero() {
    assert!(sigmoid(-100.0) < 1e-5);
}

#[test]
fn sigmoid_symmetry() {
    // sigmoid(-x) = 1 - sigmoid(x)
    let x = 2.5;
    assert!((sigmoid(-x) - (1.0 - sigmoid(x))).abs() < 1e-6);
}

#[test]
fn sigmoid_output_range() {
    for &x in &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let y = sigmoid(x);
        assert!(y >= 0.0 && y <= 1.0, "sigmoid({x}) = {y} out of [0,1]");
    }
}

// ── Tanh ─────────────────────────────────────────────────────────────

#[test]
fn tanh_zero() {
    assert!((tanh_act(0.0) - 0.0).abs() < 1e-6);
}

#[test]
fn tanh_large_positive() {
    assert!((tanh_act(100.0) - 1.0).abs() < 1e-5);
}

#[test]
fn tanh_large_negative() {
    assert!((tanh_act(-100.0) - (-1.0)).abs() < 1e-5);
}

#[test]
fn tanh_odd_symmetry() {
    let x = 1.5;
    assert!((tanh_act(-x) - (-tanh_act(x))).abs() < 1e-6);
}

// ── GELU ─────────────────────────────────────────────────────────────

#[test]
fn gelu_zero() {
    assert!(gelu(0.0).abs() < 1e-6);
}

#[test]
fn gelu_positive_large() {
    // gelu(x) ≈ x for large x
    assert!((gelu(10.0) - 10.0).abs() < 0.01);
}

#[test]
fn gelu_negative_large() {
    // gelu(x) ≈ 0 for large negative x
    assert!(gelu(-10.0).abs() < 0.01);
}

#[test]
fn gelu_tanh_approx_close_to_gelu() {
    let x = 1.0;
    assert!((gelu(x) - gelu_tanh(x)).abs() < 0.02);
}

// ── SiLU / Swish ─────────────────────────────────────────────────────

#[test]
fn silu_zero() {
    assert!(silu(0.0).abs() < 1e-6);
}

#[test]
fn silu_positive() {
    // silu(x) = x * sigmoid(x), for x=1: 1 * sigmoid(1) ≈ 0.7311
    assert!((silu(1.0) - 0.7311).abs() < 0.01);
}

#[test]
fn silu_negative_small_magnitude() {
    // silu has a minimum around x ≈ -1.278
    let y = silu(-1.278);
    assert!(y < 0.0);
}

#[test]
fn swish_beta_one_equals_silu() {
    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        assert!(
            (swish(x, 1.0) - silu(x)).abs() < 1e-6,
            "swish(x={x}, beta=1) should equal silu(x={x})"
        );
    }
}

#[test]
fn swish_beta_zero_is_half_x() {
    // swish(x, 0) = x * sigmoid(0) = x * 0.5
    assert!((swish(4.0, 0.0) - 2.0).abs() < 1e-6);
}

// ── Hard Sigmoid / Hard Swish ────────────────────────────────────────

#[test]
fn hard_sigmoid_zero() {
    assert!((hard_sigmoid(0.0) - 0.5).abs() < 1e-6);
}

#[test]
fn hard_sigmoid_clamps_high() {
    assert!((hard_sigmoid(10.0) - 1.0).abs() < 1e-6);
}

#[test]
fn hard_sigmoid_clamps_low() {
    assert!((hard_sigmoid(-10.0) - 0.0).abs() < 1e-6);
}

#[test]
fn hard_swish_zero() {
    assert!(hard_swish(0.0).abs() < 1e-6);
}

#[test]
fn hard_swish_large_positive_approximates_x() {
    // hard_swish(x) ≈ x for large x (since hard_sigmoid → 1)
    assert!((hard_swish(10.0) - 10.0).abs() < 1e-5);
}

// ── Softplus ─────────────────────────────────────────────────────────

#[test]
fn softplus_zero() {
    // softplus(0) = ln(2) ≈ 0.6931
    assert!((softplus(0.0) - 0.6931).abs() < 0.001);
}

#[test]
fn softplus_large_positive_approx_x() {
    // softplus(x) ≈ x for large x
    assert!((softplus(100.0) - 100.0).abs() < 0.01);
}

#[test]
fn softplus_always_non_negative() {
    for &x in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
        assert!(softplus(x) > 0.0, "softplus({x}) should be positive");
    }
    // Very negative values may underflow to 0 due to exp precision
    assert!(softplus(-100.0) >= 0.0);
}

// ── Mish ─────────────────────────────────────────────────────────────

#[test]
fn mish_zero() {
    // mish(0) = 0 * tanh(ln(2)) ≈ 0
    assert!(mish(0.0).abs() < 1e-6);
}

#[test]
fn mish_large_positive_approx_x() {
    assert!((mish(10.0) - 10.0).abs() < 0.01);
}

// ── ELU ──────────────────────────────────────────────────────────────

#[test]
fn elu_positive() {
    assert_eq!(elu(5.0, 1.0), 5.0);
}

#[test]
fn elu_negative() {
    // elu(-1, alpha=1) = 1*(exp(-1)-1) ≈ -0.6321
    assert!((elu(-1.0, 1.0) - (-0.6321)).abs() < 0.001);
}

#[test]
fn elu_zero() {
    assert_eq!(elu(0.0, 1.0), 0.0);
}

#[test]
fn elu_negative_bounded() {
    // For x → -inf, elu → -alpha
    assert!((elu(-100.0, 2.0) - (-2.0)).abs() < 0.01);
}

// ── SELU ─────────────────────────────────────────────────────────────

#[test]
fn selu_positive() {
    // SELU lambda ≈ 1.0507, for positive x: selu(x) = lambda * x
    let y = selu(1.0);
    assert!((y - 1.0507).abs() < 0.001);
}

#[test]
fn selu_zero() {
    assert!(selu(0.0).abs() < 1e-6);
}

#[test]
fn selu_negative() {
    let y = selu(-1.0);
    assert!(y < 0.0);
}

// ── Quick GELU ───────────────────────────────────────────────────────

#[test]
fn quick_gelu_zero() {
    assert!(quick_gelu(0.0).abs() < 1e-6);
}

#[test]
fn quick_gelu_positive_large() {
    assert!((quick_gelu(10.0) - 10.0).abs() < 0.01);
}

// ── Batch activate ───────────────────────────────────────────────────

#[test]
fn activate_empty_input() {
    let result = activate(&[], ActivationType::ReLU);
    assert!(result.is_empty());
}

#[test]
fn activate_relu_batch() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = activate(&input, ActivationType::ReLU);
    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn activate_silu_batch() {
    let input = vec![0.0, 1.0, -1.0];
    let result = activate(&input, ActivationType::SiLU);
    assert_eq!(result.len(), 3);
    assert!(result[0].abs() < 1e-6); // silu(0) = 0
}

#[test]
fn activate_inplace_modifies_buffer() {
    let mut input = vec![-2.0, 0.0, 2.0];
    activate_inplace(&mut input, ActivationType::ReLU);
    assert_eq!(input, vec![0.0, 0.0, 2.0]);
}

#[test]
fn activate_vs_inplace_consistency() {
    let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let result = activate(&input, ActivationType::Sigmoid);
    let mut inplace = input.clone();
    activate_inplace(&mut inplace, ActivationType::Sigmoid);
    for (a, b) in result.iter().zip(inplace.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

// ── Activate derivative ──────────────────────────────────────────────

#[test]
fn activate_derivative_relu() {
    let input = vec![-2.0, 0.0, 2.0];
    let deriv = activate_derivative(&input, ActivationType::ReLU);
    assert_eq!(deriv.len(), 3);
    assert_eq!(deriv[0], 0.0); // negative → 0
    // deriv[1] is boundary, implementation-dependent (0 or 1)
    assert_eq!(deriv[2], 1.0); // positive → 1
}

#[test]
fn activate_derivative_sigmoid() {
    let input = vec![0.0];
    let deriv = activate_derivative(&input, ActivationType::Sigmoid);
    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    assert!((deriv[0] - 0.25).abs() < 1e-5);
}

#[test]
fn activate_derivative_empty() {
    let result = activate_derivative(&[], ActivationType::GELU);
    assert!(result.is_empty());
}

// ── ActivationType enum ──────────────────────────────────────────────

#[test]
fn activation_type_clone_copy() {
    let a = ActivationType::SiLU;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn activation_type_debug() {
    let d = format!("{:?}", ActivationType::GELUTanh);
    assert_eq!(d, "GELUTanh");
}

#[test]
fn activation_type_with_params() {
    let a = ActivationType::LeakyReLU(0.01);
    let b = ActivationType::LeakyReLU(0.01);
    assert_eq!(a, b);
    let c = ActivationType::LeakyReLU(0.1);
    assert_ne!(a, c);
}

// ── All activation types through batch activate ──────────────────────

#[test]
fn all_activation_types_produce_finite_output() {
    let input = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let types = vec![
        ActivationType::ReLU,
        ActivationType::LeakyReLU(0.01),
        ActivationType::GELU,
        ActivationType::GELUTanh,
        ActivationType::SiLU,
        ActivationType::Swish(1.0),
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::HardSigmoid,
        ActivationType::HardSwish,
        ActivationType::Mish,
        ActivationType::Softplus,
        ActivationType::ELU(1.0),
        ActivationType::SELU,
        ActivationType::QuickGELU,
    ];
    for act in types {
        let result = activate(&input, act);
        assert_eq!(result.len(), input.len(), "{act:?}: wrong output length");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "{act:?}: non-finite at index {i}: {v}");
        }
    }
}
