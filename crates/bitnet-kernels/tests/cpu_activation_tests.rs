#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! Comprehensive tests for CPU activation functions.
//!
//! Covers mathematical accuracy (f64 reference), edge cases (NaN, Inf, zero,
//! large magnitude), output range bounds, monotonicity properties, batch
//! processing, and in-place vs out-of-place consistency.

use bitnet_kernels::cpu::activations::{
    self, ActivationType, activate, activate_derivative, activate_inplace, elu, elu_vec, gelu,
    gelu_approx_vec, gelu_inplace, gelu_tanh, gelu_vec, hard_sigmoid, hard_sigmoid_vec, hard_swish,
    hard_swish_vec, leaky_relu, leaky_relu_vec, mish, mish_vec, quick_gelu, relu, relu_inplace,
    selu, sigmoid, silu, silu_inplace, silu_vec, softplus, softplus_beta, softplus_vec, swish,
    tanh_act,
};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() < tol
}

// f64 reference implementations for numerical accuracy comparison.

fn gelu_f64(x: f64) -> f64 {
    let cdf = 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
    x * cdf
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_f64(x: f64) -> f64 {
    x * sigmoid_f64(x)
}

fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}

fn tanh_f64(x: f64) -> f64 {
    x.tanh()
}

fn softplus_f64(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn mish_f64(x: f64) -> f64 {
    x * softplus_f64(x).tanh()
}

fn elu_f64(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

const SELU_ALPHA: f64 = 1.6732632;
const SELU_LAMBDA: f64 = 1.050_701;

fn selu_f64(x: f64) -> f64 {
    SELU_LAMBDA * elu_f64(x, SELU_ALPHA)
}

fn quick_gelu_f64(x: f64) -> f64 {
    x * sigmoid_f64(1.702 * x)
}

fn hard_sigmoid_f64(x: f64) -> f64 {
    (x / 6.0 + 0.5).clamp(0.0, 1.0)
}

fn hard_swish_f64(x: f64) -> f64 {
    x * hard_sigmoid_f64(x)
}

// =====================================================================
// §1  Numerical accuracy: f32 impl vs f64 reference
// =====================================================================

#[test]
fn activations_gelu_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = gelu(x);
        let f64_val = gelu_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-5),
            "GELU accuracy: gelu({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_sigmoid_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-80..=80).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = sigmoid(x);
        let f64_val = sigmoid_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-6),
            "Sigmoid accuracy: sigmoid({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_silu_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = silu(x);
        let f64_val = silu_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-5),
            "SiLU accuracy: silu({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_relu_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = relu(x);
        let f64_val = relu_f64(x as f64) as f32;
        assert_eq!(f32_val, f64_val, "ReLU accuracy: relu({x})");
    }
}

#[test]
fn activations_tanh_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = tanh_act(x);
        let f64_val = tanh_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-6),
            "Tanh accuracy: tanh({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_softplus_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = softplus(x);
        let f64_val = softplus_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-4),
            "Softplus accuracy: softplus({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_mish_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = mish(x);
        let f64_val = mish_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-4),
            "Mish accuracy: mish({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_elu_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
    for alpha in [0.5_f32, 1.0, 2.0] {
        for &x in &points {
            let f32_val = elu(x, alpha);
            let f64_val = elu_f64(x as f64, alpha as f64) as f32;
            assert!(
                approx_eq(f32_val, f64_val, 1e-5),
                "ELU accuracy: elu({x}, {alpha}) = {f32_val}, f64 ref = {f64_val}"
            );
        }
    }
}

#[test]
fn activations_selu_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = selu(x);
        let f64_val = selu_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-4),
            "SELU accuracy: selu({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_quick_gelu_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = quick_gelu(x);
        let f64_val = quick_gelu_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-5),
            "QuickGELU accuracy: quick_gelu({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_hard_sigmoid_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = hard_sigmoid(x);
        let f64_val = hard_sigmoid_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-6),
            "HardSigmoid accuracy: hard_sigmoid({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

#[test]
fn activations_hard_swish_numerical_accuracy_vs_f64() {
    let points: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        let f32_val = hard_swish(x);
        let f64_val = hard_swish_f64(x as f64) as f32;
        assert!(
            approx_eq(f32_val, f64_val, 1e-5),
            "HardSwish accuracy: hard_swish({x}) = {f32_val}, f64 ref = {f64_val}"
        );
    }
}

// =====================================================================
// §2  Known mathematical values
// =====================================================================

#[test]
fn activations_gelu_known_values() {
    // GELU(0) = 0
    assert!(approx_eq(gelu(0.0), 0.0, 1e-7));
    // GELU(1) ≈ 0.8413 (Φ(1) ≈ 0.8413)
    assert!(approx_eq(gelu(1.0), 0.8413, 2e-4));
    // GELU(-1) ≈ -0.1587
    assert!(approx_eq(gelu(-1.0), -0.1587, 2e-4));
}

#[test]
fn activations_sigmoid_known_values() {
    assert!(approx_eq(sigmoid(0.0), 0.5, 1e-7));
    // σ(1) ≈ 0.7311
    assert!(approx_eq(sigmoid(1.0), 0.7311, 1e-4));
    // σ(-1) ≈ 0.2689
    assert!(approx_eq(sigmoid(-1.0), 0.2689, 1e-4));
}

#[test]
fn activations_silu_known_values() {
    assert!(approx_eq(silu(0.0), 0.0, 1e-7));
    // SiLU(1) = 1 * σ(1) ≈ 0.7311
    assert!(approx_eq(silu(1.0), 0.7311, 1e-4));
    // SiLU(-1) = -1 * σ(-1) ≈ -0.2689
    assert!(approx_eq(silu(-1.0), -0.2689, 1e-4));
}

#[test]
fn activations_tanh_known_values() {
    assert!(approx_eq(tanh_act(0.0), 0.0, 1e-7));
    // tanh(1) ≈ 0.7616
    assert!(approx_eq(tanh_act(1.0), 0.7616, 1e-4));
    // tanh is odd: tanh(-x) = -tanh(x)
    assert!(approx_eq(tanh_act(-1.0), -0.7616, 1e-4));
}

#[test]
fn activations_relu_known_values() {
    assert_eq!(relu(0.0), 0.0);
    assert_eq!(relu(1.0), 1.0);
    assert_eq!(relu(-1.0), 0.0);
    assert_eq!(relu(42.5), 42.5);
    assert_eq!(relu(-42.5), 0.0);
}

#[test]
fn activations_softplus_known_values() {
    // softplus(0) = ln(2) ≈ 0.6931
    assert!(approx_eq(softplus(0.0), std::f32::consts::LN_2, 1e-4));
    assert!(softplus(-5.0) > 0.0);
    assert!(softplus(5.0) > 5.0);
}

#[test]
fn activations_elu_known_values() {
    assert_eq!(elu(1.0, 1.0), 1.0);
    assert!(approx_eq(elu(0.0, 1.0), 0.0, 1e-6));
    // ELU(-inf, alpha=1) → -alpha
    assert!(approx_eq(elu(-100.0, 1.0), -1.0, 1e-4));
    assert!(approx_eq(elu(-100.0, 2.0), -2.0, 1e-4));
}

#[test]
fn activations_selu_known_values() {
    assert!(approx_eq(selu(0.0), 0.0, 1e-5));
    // SELU(1) = lambda * 1 = 1.050701
    assert!(approx_eq(selu(1.0), 1.050_701, 1e-4));
}

#[test]
fn activations_swish_known_values() {
    // Swish(x, beta=1) = SiLU(x)
    assert!(approx_eq(swish(1.0, 1.0), silu(1.0), 1e-7));
    // Swish(0, any beta) = 0
    assert!(approx_eq(swish(0.0, 5.0), 0.0, 1e-7));
}

// =====================================================================
// §3  Edge cases: zero, negative, large, NaN, infinity
// =====================================================================

#[test]
fn activations_at_negative_zero() {
    let nz = -0.0_f32;
    assert!(relu(nz) == 0.0);
    assert!(approx_eq(sigmoid(nz), 0.5, 1e-6));
    assert!(approx_eq(tanh_act(nz), 0.0, 1e-6));
    assert!(approx_eq(gelu(nz), 0.0, 1e-6));
    assert!(approx_eq(silu(nz), 0.0, 1e-6));
}

#[test]
fn activations_nan_propagation_scalar_functions() {
    let nan = f32::NAN;
    assert!(relu(nan).is_nan());
    assert!(sigmoid(nan).is_nan());
    assert!(tanh_act(nan).is_nan());
    assert!(gelu(nan).is_nan());
    assert!(gelu_tanh(nan).is_nan());
    assert!(silu(nan).is_nan());
    assert!(softplus(nan).is_nan());
    assert!(mish(nan).is_nan());
    assert!(hard_sigmoid(nan).is_nan());
    assert!(hard_swish(nan).is_nan());
    assert!(elu(nan, 1.0).is_nan());
    assert!(selu(nan).is_nan());
    assert!(quick_gelu(nan).is_nan());
    assert!(leaky_relu(nan, 0.01).is_nan());
}

#[test]
fn activations_positive_infinity_handling() {
    let inf = f32::INFINITY;
    assert_eq!(relu(inf), inf);
    assert!(approx_eq(sigmoid(inf), 1.0, 1e-6));
    assert!(approx_eq(tanh_act(inf), 1.0, 1e-6));
    assert_eq!(hard_sigmoid(inf), 1.0);
    assert!(!softplus(inf).is_nan());
}

#[test]
fn activations_negative_infinity_handling() {
    let ninf = f32::NEG_INFINITY;
    assert_eq!(relu(ninf), 0.0);
    assert!(approx_eq(sigmoid(ninf), 0.0, 1e-6));
    assert!(approx_eq(tanh_act(ninf), -1.0, 1e-6));
    assert_eq!(hard_sigmoid(ninf), 0.0);
    assert!(approx_eq(softplus(ninf), 0.0, 1e-3));
}

#[test]
fn activations_very_large_positive_no_nan() {
    let large = [1e6_f32, 1e10, 1e20, 1e38];
    for &x in &large {
        assert!(!sigmoid(x).is_nan(), "sigmoid({x}) is NaN");
        assert!(!tanh_act(x).is_nan(), "tanh({x}) is NaN");
        assert!(!softplus(x).is_nan(), "softplus({x}) is NaN");
        assert!(!hard_sigmoid(x).is_nan(), "hard_sigmoid({x}) is NaN");
        assert!(!selu(x).is_nan(), "selu({x}) is NaN");
    }
}

#[test]
fn activations_very_large_negative_no_nan() {
    let large_neg = [-1e6_f32, -1e10, -1e20, -1e38];
    for &x in &large_neg {
        assert!(!sigmoid(x).is_nan(), "sigmoid({x}) is NaN");
        assert!(!tanh_act(x).is_nan(), "tanh({x}) is NaN");
        assert!(!softplus(x).is_nan(), "softplus({x}) is NaN");
        assert!(!elu(x, 1.0).is_nan(), "elu({x}) is NaN");
        assert!(!selu(x).is_nan(), "selu({x}) is NaN");
    }
}

#[test]
fn activations_subnormal_inputs() {
    let subnormal = f32::MIN_POSITIVE / 2.0;
    assert!(subnormal > 0.0 && subnormal < f32::MIN_POSITIVE);
    assert!(!relu(subnormal).is_nan());
    assert!(!sigmoid(subnormal).is_nan());
    assert!(!gelu(subnormal).is_nan());
    assert!(!silu(subnormal).is_nan());
    assert!(!tanh_act(subnormal).is_nan());
}

// =====================================================================
// §4  Output range bounds
// =====================================================================

#[test]
fn activations_sigmoid_output_range_01() {
    let inputs: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.5).collect();
    for &x in &inputs {
        let y = sigmoid(x);
        assert!((0.0..=1.0).contains(&y), "sigmoid({x}) = {y} outside [0, 1]");
    }
}

#[test]
fn activations_tanh_output_range_neg1_pos1() {
    let inputs: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.5).collect();
    for &x in &inputs {
        let y = tanh_act(x);
        assert!((-1.0..=1.0).contains(&y), "tanh({x}) = {y} outside [-1, 1]");
    }
}

#[test]
fn activations_hard_sigmoid_output_range_01() {
    let inputs: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.5).collect();
    for &x in &inputs {
        let y = hard_sigmoid(x);
        assert!((0.0..=1.0).contains(&y), "hard_sigmoid({x}) = {y} outside [0, 1]");
    }
}

#[test]
fn activations_relu_output_nonnegative() {
    let inputs: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
    for &x in &inputs {
        assert!(relu(x) >= 0.0, "relu({x}) = {} is negative", relu(x));
    }
}

#[test]
fn activations_softplus_output_nonnegative() {
    let inputs: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
    for &x in &inputs {
        assert!(softplus(x) >= 0.0, "softplus({x}) = {} is negative", softplus(x));
    }
}

// =====================================================================
// §5  Monotonicity
// =====================================================================

fn assert_monotonic_nondecreasing(name: &str, act: ActivationType) {
    let input: Vec<f32> = (-500..=500).map(|i| i as f32 * 0.01).collect();
    let out = activate(&input, act);
    for i in 1..out.len() {
        assert!(
            out[i] >= out[i - 1] - 1e-5,
            "{name} not monotonically non-decreasing at i={i}: \
             f({}) = {} > f({}) = {}",
            input[i - 1],
            out[i - 1],
            input[i],
            out[i]
        );
    }
}

#[test]
fn activations_monotonicity_sigmoid() {
    assert_monotonic_nondecreasing("Sigmoid", ActivationType::Sigmoid);
}

#[test]
fn activations_monotonicity_tanh() {
    assert_monotonic_nondecreasing("Tanh", ActivationType::Tanh);
}

#[test]
fn activations_monotonicity_relu() {
    assert_monotonic_nondecreasing("ReLU", ActivationType::ReLU);
}

#[test]
fn activations_monotonicity_leaky_relu() {
    assert_monotonic_nondecreasing("LeakyReLU", ActivationType::LeakyReLU(0.01));
}

#[test]
fn activations_monotonicity_softplus() {
    assert_monotonic_nondecreasing("Softplus", ActivationType::Softplus);
}

#[test]
fn activations_monotonicity_elu() {
    assert_monotonic_nondecreasing("ELU", ActivationType::ELU(1.0));
}

#[test]
fn activations_monotonicity_selu() {
    assert_monotonic_nondecreasing("SELU", ActivationType::SELU);
}

#[test]
fn activations_monotonicity_hard_sigmoid() {
    assert_monotonic_nondecreasing("HardSigmoid", ActivationType::HardSigmoid);
}

// =====================================================================
// §6  Symmetry / oddness properties
// =====================================================================

#[test]
fn activations_sigmoid_symmetry_property() {
    // σ(x) + σ(-x) = 1
    let inputs: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
    for &x in &inputs {
        assert!(approx_eq(sigmoid(x) + sigmoid(-x), 1.0, 1e-5), "σ({x}) + σ({}) != 1.0", -x);
    }
}

#[test]
fn activations_tanh_odd_function() {
    // tanh(-x) = -tanh(x)
    let inputs: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
    for &x in &inputs {
        assert!(approx_eq(tanh_act(-x), -tanh_act(x), 1e-6), "tanh is not odd at x={x}");
    }
}

#[test]
fn activations_gelu_asymptotic_behavior() {
    // GELU(0) = 0
    assert!(approx_eq(gelu(0.0), 0.0, 1e-7));
    // For large positive x, GELU(x) ≈ x
    assert!(approx_eq(gelu(10.0), 10.0, 1e-3));
    // For large negative x, GELU(x) ≈ 0
    assert!(approx_eq(gelu(-10.0), 0.0, 1e-3));
}

// =====================================================================
// §7  Batch processing tests
// =====================================================================

#[test]
fn activations_batch_large_vector() {
    let n = 10_000;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 5000.0) * 0.01).collect();

    let activations = [
        ActivationType::ReLU,
        ActivationType::GELU,
        ActivationType::SiLU,
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::Softplus,
        ActivationType::Mish,
        ActivationType::ELU(1.0),
        ActivationType::SELU,
        ActivationType::QuickGELU,
        ActivationType::HardSigmoid,
        ActivationType::HardSwish,
    ];

    for act in activations {
        let out = activate(&input, act);
        assert_eq!(out.len(), n, "{act:?} output length mismatch");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{act:?} produced non-finite {v} at index {i} (input={})",
                input[i]
            );
        }
    }
}

#[test]
fn activations_batch_single_element() {
    let input = vec![1.5_f32];
    let out = activate(&input, ActivationType::GELU);
    assert_eq!(out.len(), 1);
    assert!(approx_eq(out[0], gelu(1.5), 1e-7));
}

#[test]
fn activations_batch_empty() {
    let input: Vec<f32> = vec![];
    for act in
        [ActivationType::ReLU, ActivationType::GELU, ActivationType::SiLU, ActivationType::Sigmoid]
    {
        assert!(activate(&input, act).is_empty());
    }
}

#[test]
fn activations_vec_functions_large_batch() {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();

    assert_eq!(gelu_vec(&input).len(), n);
    assert_eq!(gelu_approx_vec(&input).len(), n);
    assert_eq!(silu_vec(&input).len(), n);
    assert_eq!(mish_vec(&input).len(), n);
    assert_eq!(softplus_vec(&input, 1.0).len(), n);
    assert_eq!(hard_sigmoid_vec(&input).len(), n);
    assert_eq!(hard_swish_vec(&input).len(), n);
    assert_eq!(leaky_relu_vec(&input, 0.01).len(), n);
    assert_eq!(elu_vec(&input, 1.0).len(), n);
}

// =====================================================================
// §8  In-place vs out-of-place consistency
// =====================================================================

#[test]
fn activations_gelu_inplace_vs_allocating() {
    let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
    let expected = gelu_vec(&input);
    let mut buf = input.clone();
    gelu_inplace(&mut buf);
    for (i, (&a, &b)) in buf.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(a, b, 1e-7), "gelu_inplace mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn activations_silu_inplace_vs_allocating() {
    let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
    let expected = silu_vec(&input);
    let mut buf = input.clone();
    silu_inplace(&mut buf);
    for (i, (&a, &b)) in buf.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(a, b, 1e-7), "silu_inplace mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn activations_relu_inplace_vs_allocating() {
    let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
    let expected = activate(&input, ActivationType::ReLU);
    let mut buf = input.clone();
    relu_inplace(&mut buf);
    assert_eq!(buf, expected);
}

#[test]
fn activations_inplace_vs_allocating_all_types() {
    let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.3).collect();
    let all_types = [
        ActivationType::ReLU,
        ActivationType::LeakyReLU(0.01),
        ActivationType::GELU,
        ActivationType::GELUTanh,
        ActivationType::SiLU,
        ActivationType::Swish(1.5),
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
    for act in all_types {
        let expected = activate(&input, act);
        let mut buf = input.clone();
        activate_inplace(&mut buf, act);
        for (i, (&a, &b)) in buf.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-7), "{act:?} inplace mismatch at {i}: {a} vs {b}");
        }
    }
}

// =====================================================================
// §9  Derivative consistency checks
// =====================================================================

#[test]
fn activations_derivative_gelu_at_zero() {
    // GELU'(0) = Φ(0) + 0·φ(0) = 0.5
    let d = activate_derivative(&[0.0], ActivationType::GELU);
    assert!(approx_eq(d[0], 0.5, 1e-4));
}

#[test]
fn activations_derivative_silu_at_zero() {
    // SiLU'(0) = σ(0) + 0·σ(0)·(1-σ(0)) = 0.5
    let d = activate_derivative(&[0.0], ActivationType::SiLU);
    assert!(approx_eq(d[0], 0.5, 1e-5));
}

#[test]
fn activations_derivative_elu_continuity_at_zero() {
    // For alpha=1, both sides of 0 have derivative = 1
    let d_pos = activate_derivative(&[1e-6], ActivationType::ELU(1.0));
    let d_neg = activate_derivative(&[-1e-6], ActivationType::ELU(1.0));
    assert!(approx_eq(d_pos[0], 1.0, 1e-3));
    assert!(approx_eq(d_neg[0], 1.0, 1e-3));
}

// =====================================================================
// §10  Relationship tests between activations
// =====================================================================

#[test]
fn activations_swish_beta1_equals_silu() {
    let points: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        assert!(approx_eq(swish(x, 1.0), silu(x), 1e-7), "Swish(β=1) != SiLU at x={x}");
    }
}

#[test]
fn activations_softplus_deriv_equals_sigmoid() {
    // d/dx softplus(x) = sigmoid(x)
    let points: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
    let deriv = activate_derivative(&points, ActivationType::Softplus);
    let sig = activate(&points, ActivationType::Sigmoid);
    for (i, (&d, &s)) in deriv.iter().zip(sig.iter()).enumerate() {
        assert!(
            approx_eq(d, s, 1e-5),
            "softplus'({}) = {d} != sigmoid({}) = {s}",
            points[i],
            points[i]
        );
    }
}

#[test]
fn activations_gelu_tanh_approximates_gelu_erf() {
    let points: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
    for &x in &points {
        assert!(
            approx_eq(gelu_tanh(x), gelu(x), 0.02),
            "GELUTanh too far from GELU at x={x}: {} vs {}",
            gelu_tanh(x),
            gelu(x)
        );
    }
}

#[test]
fn activations_elu_alpha_zero_equals_relu() {
    let points: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
    for &x in &points {
        assert!(approx_eq(elu(x, 0.0), relu(x), 1e-7), "ELU(α=0) != ReLU at x={x}");
    }
}

#[test]
fn activations_hard_swish_zero_for_very_negative() {
    for x in [-3.0, -4.0, -10.0, -100.0] {
        assert_eq!(hard_swish(x), 0.0, "hard_swish({x}) should be 0");
    }
}

#[test]
fn activations_hard_swish_equals_x_for_very_positive() {
    for x in [3.0, 4.0, 10.0, 100.0] {
        assert!(approx_eq(hard_swish(x), x, 1e-5), "hard_swish({x}) should equal {x}");
    }
}

// =====================================================================
// §11  Property tests (proptest)
// =====================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn activations_prop_sigmoid_in_01(x in -100.0_f32..100.0) {
        let y = sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&y), "sigmoid({x}) = {y} outside [0,1]");
    }

    #[test]
    fn activations_prop_tanh_in_neg1_pos1(x in -100.0_f32..100.0) {
        let y = tanh_act(x);
        prop_assert!((-1.0..=1.0).contains(&y), "tanh({x}) = {y} outside [-1,1]");
    }

    #[test]
    fn activations_prop_hard_sigmoid_in_01(x in -1000.0_f32..1000.0) {
        let y = hard_sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&y), "hard_sigmoid({x}) = {y} outside [0,1]");
    }

    #[test]
    fn activations_prop_relu_nonnegative(x in -1000.0_f32..1000.0) {
        prop_assert!(relu(x) >= 0.0, "relu({x}) = {} is negative", relu(x));
    }

    #[test]
    fn activations_prop_softplus_nonnegative(x in -50.0_f32..50.0) {
        prop_assert!(softplus(x) >= 0.0, "softplus({x}) = {} is negative", softplus(x));
    }

    #[test]
    fn activations_prop_sigmoid_symmetry(x in -50.0_f32..50.0) {
        let sum = sigmoid(x) + sigmoid(-x);
        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "σ({x}) + σ({}) = {sum}, expected 1.0", -x
        );
    }

    #[test]
    fn activations_prop_tanh_odd(x in -50.0_f32..50.0) {
        let diff = (tanh_act(-x) - (-tanh_act(x))).abs();
        prop_assert!(diff < 1e-6, "tanh not odd at x={x}, diff={diff}");
    }

    #[test]
    fn activations_prop_relu_identity_positive(x in 0.0_f32..1e6) {
        prop_assert!((relu(x) - x).abs() < f32::EPSILON, "relu({}) != {} for positive input", x, x);
    }

    #[test]
    fn activations_prop_relu_zero_negative(x in -1e6_f32..0.0) {
        prop_assert!(relu(x) == 0.0, "relu({x}) = {} should be 0", relu(x));
    }

    #[test]
    fn activations_prop_sigmoid_monotonic(
        x1 in -50.0_f32..50.0,
        delta in 0.0_f32..10.0,
    ) {
        let x2 = x1 + delta;
        prop_assert!(
            sigmoid(x2) >= sigmoid(x1) - 1e-6,
            "sigmoid not monotonic: σ({x1})={} > σ({x2})={}",
            sigmoid(x1), sigmoid(x2)
        );
    }

    #[test]
    fn activations_prop_softplus_monotonic(
        x1 in -50.0_f32..50.0,
        delta in 0.0_f32..10.0,
    ) {
        let x2 = x1 + delta;
        prop_assert!(
            softplus(x2) >= softplus(x1) - 1e-5,
            "softplus not monotonic: sp({x1})={} > sp({x2})={}",
            softplus(x1), softplus(x2)
        );
    }

    #[test]
    fn activations_prop_elu_monotonic(
        x1 in -50.0_f32..50.0,
        delta in 0.0_f32..10.0,
    ) {
        let x2 = x1 + delta;
        prop_assert!(
            elu(x2, 1.0) >= elu(x1, 1.0) - 1e-5,
            "ELU not monotonic: elu({x1})={} > elu({x2})={}",
            elu(x1, 1.0), elu(x2, 1.0)
        );
    }

    #[test]
    fn activations_prop_gelu_accuracy_f64(x in -10.0_f32..10.0) {
        let f32_val = gelu(x);
        let f64_val = gelu_f64(x as f64) as f32;
        prop_assert!(
            (f32_val - f64_val).abs() < 1e-5,
            "GELU accuracy: gelu({x}) = {f32_val}, f64 = {f64_val}"
        );
    }

    #[test]
    fn activations_prop_silu_accuracy_f64(x in -10.0_f32..10.0) {
        let f32_val = silu(x);
        let f64_val = silu_f64(x as f64) as f32;
        prop_assert!(
            (f32_val - f64_val).abs() < 1e-5,
            "SiLU accuracy: silu({x}) = {f32_val}, f64 = {f64_val}"
        );
    }

    #[test]
    fn activations_prop_inplace_matches_allocating(
        x in proptest::collection::vec(-10.0_f32..10.0, 1..64),
    ) {
        let expected = activate(&x, ActivationType::GELU);
        let mut buf = x;
        activate_inplace(&mut buf, ActivationType::GELU);
        for (i, (&a, &b)) in buf.iter().zip(expected.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-7,
                "inplace mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn activations_prop_swish_beta1_is_silu(x in -20.0_f32..20.0) {
        let diff = (swish(x, 1.0) - silu(x)).abs();
        prop_assert!(diff < 1e-7, "Swish(β=1) != SiLU at x={x}, diff={diff}");
    }

    #[test]
    fn activations_prop_no_nan_finite_input(x in -1e10_f32..1e10) {
        let acts = [
            ActivationType::ReLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::HardSigmoid,
            ActivationType::HardSwish,
            ActivationType::Softplus,
            ActivationType::SELU,
        ];
        for act in &acts {
            let out = activate(&[x], *act);
            prop_assert!(
                !out[0].is_nan(),
                "{act:?} produced NaN for finite input {x}"
            );
        }
    }

    #[test]
    fn activations_prop_leaky_relu_negative_slope(
        x in -100.0_f32..0.0,
        alpha in 0.001_f32..0.5,
    ) {
        let y = leaky_relu(x, alpha);
        prop_assert!(
            y < 0.0,
            "leaky_relu({x}, {alpha}) = {y} should be negative"
        );
        prop_assert!(
            (y - alpha * x).abs() < 1e-4,
            "leaky_relu({x}, {alpha}) = {y}, expected {}",
            alpha * x
        );
    }

    #[test]
    fn activations_prop_softplus_beta_nonneg(
        x in -10.0_f32..10.0,
        beta in 0.1_f32..10.0,
    ) {
        let y = softplus_beta(x, beta);
        prop_assert!(!y.is_nan(), "softplus_beta({x}, {beta}) is NaN");
        prop_assert!(y >= 0.0, "softplus_beta({x}, {beta}) = {y} is negative");
    }
}

// =====================================================================
// §12  Dispatch consistency: apply_activation = activate
// =====================================================================

#[test]
fn activations_apply_activation_equals_activate() {
    let input: Vec<f32> = (-15..=15).map(|i| i as f32 * 0.2).collect();
    let types = [
        ActivationType::ReLU,
        ActivationType::LeakyReLU(0.01),
        ActivationType::GELU,
        ActivationType::GELUTanh,
        ActivationType::SiLU,
        ActivationType::Swish(2.0),
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
        let a = activate(&input, act);
        let b = activations::apply_activation(&input, act);
        assert_eq!(a, b, "apply_activation != activate for {act:?}");
    }
}
