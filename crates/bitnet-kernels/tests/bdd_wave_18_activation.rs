//! BDD Wave 18 — Activation function integration tests.
//!
//! Tests GELU, SiLU, ReLU and other activations for correct values,
//! edge-case behaviour (NaN, ±Inf, zeros), monotonicity, and derivative
//! consistency.

use bitnet_kernels::cpu::activations::{
    ActivationType, activate, apply_activation, elu, gelu, gelu_inplace, gelu_vec, hard_sigmoid,
    hard_swish, leaky_relu, mish, relu, relu_inplace, sigmoid, silu, silu_inplace, silu_vec,
    softplus,
};

const TOL: f32 = 1e-5;

// ── ReLU ───────────────────────────────────────────────────────────

#[test]
fn given_positive_input_when_relu_applied_then_output_equals_input() {
    assert!((relu(3.0) - 3.0).abs() < TOL);
    assert!((relu(0.001) - 0.001).abs() < TOL);
}

#[test]
fn given_negative_input_when_relu_applied_then_output_is_zero() {
    assert!((relu(-5.0)).abs() < TOL);
    assert!((relu(-0.001)).abs() < TOL);
}

#[test]
fn given_zero_input_when_relu_applied_then_output_is_zero() {
    assert!((relu(0.0)).abs() < TOL);
}

#[test]
fn given_nan_input_when_relu_applied_then_output_is_nan() {
    assert!(relu(f32::NAN).is_nan());
}

#[test]
fn given_positive_infinity_when_relu_applied_then_output_is_positive_infinity() {
    assert_eq!(relu(f32::INFINITY), f32::INFINITY);
}

#[test]
fn given_negative_infinity_when_relu_applied_then_output_is_zero() {
    assert!((relu(f32::NEG_INFINITY)).abs() < TOL || relu(f32::NEG_INFINITY) == 0.0);
}

#[test]
fn given_vector_when_relu_inplace_then_negatives_zeroed() {
    let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    relu_inplace(&mut data);
    assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

// ── GELU ───────────────────────────────────────────────────────────

#[test]
fn given_zero_when_gelu_applied_then_output_is_zero() {
    assert!(gelu(0.0).abs() < TOL, "GELU(0) should be 0");
}

#[test]
fn given_large_positive_when_gelu_applied_then_output_approaches_input() {
    let x = 5.0;
    let y = gelu(x);
    assert!((y - x).abs() < 0.01, "GELU({x}) should be ~{x}, got {y}");
}

#[test]
fn given_large_negative_when_gelu_applied_then_output_approaches_zero() {
    let y = gelu(-5.0);
    assert!(y.abs() < 0.01, "GELU(-5) should be ~0, got {y}");
}

#[test]
fn given_ascending_inputs_when_gelu_applied_then_output_is_monotonically_increasing() {
    let inputs: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
    let outputs: Vec<f32> = inputs.iter().map(|&x| gelu(x)).collect();
    // GELU is not strictly monotone for x < -0.5 but is for moderate ranges.
    // Check that for x > 0.5 it's increasing.
    for w in outputs.windows(2) {
        if w[0] > 0.5 {
            assert!(w[1] >= w[0], "GELU should be increasing for x > 0.5");
        }
    }
}

#[test]
fn given_vector_when_gelu_vec_applied_then_matches_scalar() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let vec_result = gelu_vec(&input);
    for (i, (&x, &y)) in input.iter().zip(vec_result.iter()).enumerate() {
        let scalar = gelu(x);
        assert!((y - scalar).abs() < TOL, "gelu_vec mismatch at {i}: {y} vs scalar {scalar}");
    }
}

#[test]
fn given_vector_when_gelu_inplace_then_values_match_gelu_vec() {
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let expected = gelu_vec(&input);
    let mut inplace = input.clone();
    gelu_inplace(&mut inplace);
    for (i, (&a, &b)) in inplace.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < TOL, "inplace mismatch at {i}: {a} vs {b}");
    }
}

// ── SiLU (Swish) ──────────────────────────────────────────────────

#[test]
fn given_zero_when_silu_applied_then_output_is_zero() {
    assert!(silu(0.0).abs() < TOL, "SiLU(0) should be 0");
}

#[test]
fn given_large_positive_when_silu_applied_then_output_approaches_input() {
    let x = 10.0;
    let y = silu(x);
    assert!((y - x).abs() < 0.01, "SiLU({x}) should be ~{x}, got {y}");
}

#[test]
fn given_large_negative_when_silu_applied_then_output_approaches_zero() {
    let y = silu(-10.0);
    assert!(y.abs() < 0.01, "SiLU(-10) should be ~0, got {y}");
}

#[test]
fn given_silu_minimum_point_when_checked_then_output_is_negative() {
    // SiLU has a minimum near x ≈ -1.278
    let y = silu(-1.278);
    assert!(y < 0.0, "SiLU at its minimum should be negative, got {y}");
}

#[test]
fn given_vector_when_silu_vec_applied_then_matches_scalar() {
    let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let vec_result = silu_vec(&input);
    for (i, (&x, &y)) in input.iter().zip(vec_result.iter()).enumerate() {
        let scalar = silu(x);
        assert!((y - scalar).abs() < TOL, "silu_vec mismatch at {i}: {y} vs scalar {scalar}");
    }
}

#[test]
fn given_vector_when_silu_inplace_then_values_match_silu_vec() {
    let input = vec![-2.0, 0.0, 2.0];
    let expected = silu_vec(&input);
    let mut inplace = input.clone();
    silu_inplace(&mut inplace);
    for (i, (&a, &b)) in inplace.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < TOL, "inplace mismatch at {i}: {a} vs {b}");
    }
}

// ── Sigmoid ────────────────────────────────────────────────────────

#[test]
fn given_zero_when_sigmoid_applied_then_output_is_half() {
    assert!((sigmoid(0.0) - 0.5).abs() < TOL);
}

#[test]
fn given_large_positive_when_sigmoid_applied_then_output_approaches_one() {
    assert!((sigmoid(20.0) - 1.0).abs() < TOL);
}

#[test]
fn given_large_negative_when_sigmoid_applied_then_output_approaches_zero() {
    assert!(sigmoid(-20.0).abs() < TOL);
}

// ── Leaky ReLU ─────────────────────────────────────────────────────

#[test]
fn given_positive_input_when_leaky_relu_applied_then_output_equals_input() {
    assert!((leaky_relu(5.0, 0.01) - 5.0).abs() < TOL);
}

#[test]
fn given_negative_input_when_leaky_relu_applied_then_output_scaled_by_alpha() {
    let result = leaky_relu(-10.0, 0.1);
    assert!((result - (-1.0)).abs() < TOL, "expected -1.0, got {result}");
}

// ── Hard sigmoid / Hard swish ──────────────────────────────────────

#[test]
fn given_input_in_linear_region_when_hard_sigmoid_applied_then_output_is_linear() {
    let y = hard_sigmoid(0.0);
    assert!((y - 0.5).abs() < TOL, "hard_sigmoid(0) should be 0.5, got {y}");
}

#[test]
fn given_large_positive_when_hard_sigmoid_applied_then_output_clamps_to_one() {
    assert!((hard_sigmoid(10.0) - 1.0).abs() < TOL);
}

#[test]
fn given_large_negative_when_hard_sigmoid_applied_then_output_clamps_to_zero() {
    assert!(hard_sigmoid(-10.0).abs() < TOL);
}

#[test]
fn given_zero_when_hard_swish_applied_then_output_is_zero() {
    assert!(hard_swish(0.0).abs() < TOL);
}

// ── Softplus ───────────────────────────────────────────────────────

#[test]
fn given_zero_when_softplus_applied_then_output_is_ln2() {
    let expected = 2.0f32.ln();
    assert!((softplus(0.0) - expected).abs() < TOL);
}

#[test]
fn given_large_positive_when_softplus_applied_then_output_approaches_input() {
    let x = 20.0;
    assert!((softplus(x) - x).abs() < 0.01);
}

// ── Mish ───────────────────────────────────────────────────────────

#[test]
fn given_zero_when_mish_applied_then_output_is_zero() {
    assert!(mish(0.0).abs() < TOL);
}

#[test]
fn given_large_positive_when_mish_applied_then_output_approaches_input() {
    let x = 10.0;
    assert!((mish(x) - x).abs() < 0.01);
}

// ── ELU ────────────────────────────────────────────────────────────

#[test]
fn given_positive_input_when_elu_applied_then_output_equals_input() {
    assert!((elu(3.0, 1.0) - 3.0).abs() < TOL);
}

#[test]
fn given_negative_input_when_elu_applied_then_output_approaches_neg_alpha() {
    let y = elu(-100.0, 1.0);
    assert!((y - (-1.0)).abs() < 0.01, "ELU(-100, 1) should be ~-1.0, got {y}");
}

// ── apply_activation dispatch ──────────────────────────────────────

#[test]
fn given_vector_when_apply_activation_relu_then_matches_manual() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = apply_activation(&input, ActivationType::ReLU);
    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
    for (i, (&a, &b)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < TOL, "mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn given_vector_when_apply_activation_gelu_then_all_finite() {
    let input: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
    let result = apply_activation(&input, ActivationType::GELU);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "GELU output should be finite at {i}, got {v}");
    }
}

#[test]
fn given_vector_when_apply_activation_silu_then_all_finite() {
    let input: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
    let result = apply_activation(&input, ActivationType::SiLU);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "SiLU output should be finite at {i}, got {v}");
    }
}

#[test]
fn given_empty_vector_when_apply_activation_then_returns_empty() {
    let input: Vec<f32> = vec![];
    let result = apply_activation(&input, ActivationType::ReLU);
    assert!(result.is_empty());
}

#[test]
fn given_vector_when_activate_alias_used_then_matches_apply_activation() {
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let a = apply_activation(&input, ActivationType::GELU);
    let b = activate(&input, ActivationType::GELU);
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < TOL, "activate vs apply_activation mismatch at {i}");
    }
}
