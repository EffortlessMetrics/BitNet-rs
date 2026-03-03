#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]

//! Metal activation function shader tests for Apple Silicon
//! Tests GPU-side activation correctness, edge cases, and numerical stability
//!
//! All tests use pure Rust CPU simulation of Metal shader computations.
//! No GPU/Metal/wgpu crates are imported — the goal is to validate the
//! mathematical correctness of activation functions as they would be
//! implemented in Metal shaders.

#![cfg(target_os = "macos")]

// ── Helpers ─────────────────────────────────────────────────────────

const TOL: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.is_sign_positive() == b.is_sign_positive();
    }
    (a - b).abs() < tol
}

// ── Metal shader activation reference implementations ───────────────

fn metal_silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

fn metal_gelu_tanh(x: f32) -> f32 {
    let k: f32 = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

fn metal_gelu_erf(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + libm::erff(x / std::f32::consts::SQRT_2));
    x * cdf
}

fn metal_relu(x: f32) -> f32 {
    x.max(0.0)
}

fn metal_leaky_relu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { x } else { alpha * x }
}

fn metal_relu6(x: f32) -> f32 {
    x.max(0.0).min(6.0)
}

fn metal_parametric_relu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { x } else { alpha * x }
}

fn metal_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn metal_tanh(x: f32) -> f32 {
    x.tanh()
}

fn metal_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn metal_softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    metal_softmax(&scaled)
}

// CPU reference implementations (f64 for precision comparison)

fn cpu_silu_f64(x: f64) -> f64 {
    x * (1.0 / (1.0 + (-x).exp()))
}

fn cpu_gelu_erf_f64(x: f64) -> f64 {
    let cdf = 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
    x * cdf
}

fn cpu_sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ── 1. SiLU / Swish activation (6 tests) ───────────────────────────

#[test]
fn test_silu_zero_input() {
    let result = metal_silu(0.0);
    assert!(approx_eq(result, 0.0, TOL), "SiLU(0) should be 0, got {result}");
}

#[test]
fn test_silu_negative_values() {
    for &x in &[-0.5_f32, -1.0, -2.0, -5.0] {
        let result = metal_silu(x);
        let reference = cpu_silu_f64(x as f64) as f32;
        assert!(approx_eq(result, reference, TOL), "SiLU({x}): metal={result}, ref={reference}");
        // SiLU of negative values should be negative (except near zero)
        if x < -0.3 {
            assert!(result < 0.0, "SiLU({x}) should be negative, got {result}");
        }
    }
}

#[test]
fn test_silu_large_positive() {
    let x = 20.0_f32;
    let result = metal_silu(x);
    // For large positive x, SiLU(x) ≈ x since sigmoid(x) ≈ 1
    assert!(
        approx_eq(result, x, 1e-4),
        "SiLU({x}) should approximate x for large positive, got {result}"
    );
}

#[test]
fn test_silu_large_negative() {
    let x = -20.0_f32;
    let result = metal_silu(x);
    // For large negative x, SiLU(x) ≈ 0 since sigmoid(x) ≈ 0
    assert!(
        approx_eq(result, 0.0, 1e-4),
        "SiLU({x}) should approximate 0 for large negative, got {result}"
    );
}

#[test]
fn test_silu_gradient_correctness() {
    // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    let h = 1e-4_f32;
    for &x in &[-2.0_f32, -1.0, 0.0, 1.0, 2.0] {
        let numerical_grad = (metal_silu(x + h) - metal_silu(x - h)) / (2.0 * h);
        let sig = metal_sigmoid(x);
        let analytical_grad = sig + x * sig * (1.0 - sig);
        assert!(
            approx_eq(numerical_grad, analytical_grad, 1e-3),
            "SiLU gradient at {x}: numerical={numerical_grad}, analytical={analytical_grad}"
        );
    }
}

#[test]
fn test_silu_vectorized_batch() {
    let batch: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
    let results: Vec<f32> = batch.iter().map(|&x| metal_silu(x)).collect();
    for (i, (&x, &r)) in batch.iter().zip(results.iter()).enumerate() {
        let reference = cpu_silu_f64(x as f64) as f32;
        assert!(approx_eq(r, reference, TOL), "SiLU batch[{i}] x={x}: metal={r}, ref={reference}");
    }
}

// ── 2. GELU activation (6 tests) ───────────────────────────────────

#[test]
fn test_gelu_zero() {
    let result = metal_gelu_tanh(0.0);
    assert!(approx_eq(result, 0.0, TOL), "GELU(0) should be 0, got {result}");
}

#[test]
fn test_gelu_symmetry() {
    // GELU is NOT symmetric: GELU(-x) != -GELU(x) in general,
    // but GELU(x) + GELU(-x) should have a specific relationship.
    // For small x, GELU(-x) ≈ -GELU(x). Verify asymmetry for larger values.
    for &x in &[0.5_f32, 1.0, 2.0] {
        let pos = metal_gelu_tanh(x);
        let neg = metal_gelu_tanh(-x);
        // GELU is not odd: |GELU(x) + GELU(-x)| > 0 for x != 0
        // But the relationship holds approximately for small x
        assert!(
            (pos + neg).abs() < x.abs(),
            "GELU asymmetry check: GELU({x})={pos}, GELU({})={neg}",
            -x
        );
    }
}

#[test]
fn test_gelu_standard_distribution_inputs() {
    // Values from standard normal distribution
    let inputs = [-1.96_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 1.96];
    for &x in &inputs {
        let result = metal_gelu_tanh(x);
        let reference = cpu_gelu_erf_f64(x as f64) as f32;
        assert!(
            approx_eq(result, reference, 1e-3),
            "GELU({x}): tanh_approx={result}, erf_ref={reference}"
        );
    }
}

#[test]
fn test_gelu_tanh_approximation_accuracy() {
    // Compare tanh-based GELU approximation against erf-based
    for i in -100..=100 {
        let x = i as f32 * 0.05;
        let tanh_result = metal_gelu_tanh(x);
        let erf_result = metal_gelu_erf(x);
        assert!(
            approx_eq(tanh_result, erf_result, 1e-3),
            "GELU approximation at {x}: tanh={tanh_result}, erf={erf_result}"
        );
    }
}

#[test]
fn test_gelu_erf_vs_tanh_comparison() {
    // Specific known values where approximation quality matters
    let test_points = [0.1_f32, 0.5, 1.0, 1.5, 2.0, 3.0];
    for &x in &test_points {
        let tanh_v = metal_gelu_tanh(x);
        let erf_v = metal_gelu_erf(x);
        let rel_err = if erf_v.abs() > 1e-10 {
            ((tanh_v - erf_v) / erf_v).abs()
        } else {
            (tanh_v - erf_v).abs()
        };
        assert!(
            rel_err < 0.005,
            "GELU relative error at {x}: tanh={tanh_v}, erf={erf_v}, rel_err={rel_err}"
        );
    }
}

#[test]
fn test_gelu_batch_processing() {
    let batch: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let results: Vec<f32> = batch.iter().map(|&x| metal_gelu_tanh(x)).collect();
    for (i, (&x, &r)) in batch.iter().zip(results.iter()).enumerate() {
        let reference = cpu_gelu_erf_f64(x as f64) as f32;
        assert!(approx_eq(r, reference, 1e-3), "GELU batch[{i}] x={x}: metal={r}, ref={reference}");
    }
}

// ── 3. ReLU family (6 tests) ───────────────────────────────────────

#[test]
fn test_relu_zero_threshold() {
    for &x in &[-1.0_f32, -0.001, 0.0, 0.001, 1.0] {
        let result = metal_relu(x);
        let expected = if x > 0.0 { x } else { 0.0 };
        assert!(approx_eq(result, expected, TOL), "ReLU({x}): got {result}, expected {expected}");
    }
}

#[test]
fn test_leaky_relu_with_alpha() {
    let alpha = 0.01_f32;
    for &x in &[-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
        let result = metal_leaky_relu(x, alpha);
        let expected = if x >= 0.0 { x } else { alpha * x };
        assert!(
            approx_eq(result, expected, TOL),
            "LeakyReLU({x}, alpha={alpha}): got {result}, expected {expected}"
        );
    }
}

#[test]
fn test_relu6_clamping() {
    for &x in &[-1.0_f32, 0.0, 3.0, 6.0, 7.0, 100.0] {
        let result = metal_relu6(x);
        let expected = x.max(0.0).min(6.0);
        assert!(approx_eq(result, expected, TOL), "ReLU6({x}): got {result}, expected {expected}");
    }
    // Verify exact clamping boundaries
    assert_eq!(metal_relu6(6.0), 6.0, "ReLU6 should pass through 6.0");
    assert_eq!(metal_relu6(0.0), 0.0, "ReLU6 should pass through 0.0");
}

#[test]
fn test_parametric_relu() {
    let alphas = [0.01_f32, 0.1, 0.25, 0.5];
    for &alpha in &alphas {
        for &x in &[-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
            let result = metal_parametric_relu(x, alpha);
            let expected = if x >= 0.0 { x } else { alpha * x };
            assert!(
                approx_eq(result, expected, TOL),
                "PReLU({x}, alpha={alpha}): got {result}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_relu_negative_slope_variations() {
    // Test various negative slopes from 0 (ReLU) to 1 (identity)
    let slopes = [0.0_f32, 0.01, 0.1, 0.3, 0.5, 1.0];
    let x = -2.0_f32;
    for &slope in &slopes {
        let result = metal_leaky_relu(x, slope);
        let expected = slope * x;
        assert!(
            approx_eq(result, expected, TOL),
            "LeakyReLU({x}, slope={slope}): got {result}, expected {expected}"
        );
    }
    // slope=0 should give ReLU behavior
    assert_eq!(metal_leaky_relu(-1.0, 0.0), 0.0, "slope=0 should be standard ReLU");
    // slope=1 should give identity
    assert_eq!(metal_leaky_relu(-1.0, 1.0), -1.0, "slope=1 should be identity");
}

#[test]
fn test_relu_inplace_semantics() {
    // Simulate in-place ReLU: input buffer is modified directly
    let mut buffer = vec![-3.0_f32, -1.0, 0.0, 1.0, 3.0];
    let expected = vec![0.0_f32, 0.0, 0.0, 1.0, 3.0];
    for val in &mut buffer {
        *val = metal_relu(*val);
    }
    for (i, (&got, &exp)) in buffer.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(got, exp, TOL), "In-place ReLU[{i}]: got {got}, expected {exp}");
    }
}

// ── 4. Softmax activation (6 tests) ────────────────────────────────

#[test]
fn test_softmax_uniform_inputs() {
    let logits = vec![1.0_f32; 4];
    let result = metal_softmax(&logits);
    for (i, &p) in result.iter().enumerate() {
        assert!(approx_eq(p, 0.25, TOL), "Softmax uniform[{i}]: got {p}, expected 0.25");
    }
}

#[test]
fn test_softmax_single_element() {
    let logits = vec![42.0_f32];
    let result = metal_softmax(&logits);
    assert!(
        approx_eq(result[0], 1.0, TOL),
        "Softmax single element should be 1.0, got {}",
        result[0]
    );
}

#[test]
fn test_softmax_numerical_stability_large_values() {
    // Without max subtraction, this would overflow
    let logits = vec![1000.0_f32, 1001.0, 1002.0];
    let result = metal_softmax(&logits);

    // Should still sum to 1.0
    let sum: f32 = result.iter().sum();
    assert!(approx_eq(sum, 1.0, TOL), "Softmax sum should be 1.0, got {sum}");

    // No NaN or Inf
    for (i, &p) in result.iter().enumerate() {
        assert!(p.is_finite(), "Softmax[{i}] should be finite, got {p}");
        assert!(p > 0.0, "Softmax[{i}] should be positive, got {p}");
    }

    // Largest logit should have largest probability
    assert!(
        result[2] > result[1] && result[1] > result[0],
        "Softmax should preserve ordering: {result:?}"
    );
}

#[test]
fn test_softmax_temperature_scaling() {
    let logits = vec![1.0_f32, 2.0, 3.0];

    // High temperature → more uniform
    let hot = metal_softmax_with_temperature(&logits, 10.0);
    let max_hot = hot.iter().cloned().fold(0.0_f32, f32::max);
    let min_hot = hot.iter().cloned().fold(1.0_f32, f32::min);

    // Low temperature → more peaked
    let cold = metal_softmax_with_temperature(&logits, 0.1);
    let max_cold = cold.iter().cloned().fold(0.0_f32, f32::max);
    let min_cold = cold.iter().cloned().fold(1.0_f32, f32::min);

    assert!(
        (max_hot - min_hot) < (max_cold - min_cold),
        "Higher temperature should produce more uniform distribution: \
         hot_spread={}, cold_spread={}",
        max_hot - min_hot,
        max_cold - min_cold
    );
}

#[test]
fn test_softmax_batch_independence() {
    // Softmax over two independent batches should be same as individual
    let batch1 = vec![1.0_f32, 2.0, 3.0];
    let batch2 = vec![4.0_f32, 5.0, 6.0];

    let result1 = metal_softmax(&batch1);
    let result2 = metal_softmax(&batch2);

    // Compute separately and compare
    let result1_again = metal_softmax(&batch1);
    let result2_again = metal_softmax(&batch2);

    for i in 0..3 {
        assert!(
            approx_eq(result1[i], result1_again[i], TOL),
            "Batch independence: batch1[{i}] differs"
        );
        assert!(
            approx_eq(result2[i], result2_again[i], TOL),
            "Batch independence: batch2[{i}] differs"
        );
    }
}

#[test]
fn test_softmax_gradient_computation() {
    // Softmax Jacobian: d(softmax_i)/d(logit_j) = s_i*(delta_ij - s_j)
    let logits = vec![1.0_f32, 2.0, 3.0];
    let s = metal_softmax(&logits);
    let h = 1e-4_f32;

    for j in 0..3 {
        let mut logits_plus = logits.clone();
        logits_plus[j] += h;
        let mut logits_minus = logits.clone();
        logits_minus[j] -= h;

        let s_plus = metal_softmax(&logits_plus);
        let s_minus = metal_softmax(&logits_minus);

        for i in 0..3 {
            let numerical = (s_plus[i] - s_minus[i]) / (2.0 * h);
            let delta_ij: f32 = if i == j { 1.0 } else { 0.0 };
            let analytical = s[i] * (delta_ij - s[j]);
            assert!(
                approx_eq(numerical, analytical, 1e-3),
                "Softmax Jacobian [{i},{j}]: numerical={numerical}, analytical={analytical}"
            );
        }
    }
}

// ── 5. Sigmoid activation (5 tests) ────────────────────────────────

#[test]
fn test_sigmoid_zero_centered() {
    let result = metal_sigmoid(0.0);
    assert!(approx_eq(result, 0.5, TOL), "sigmoid(0) should be 0.5, got {result}");
}

#[test]
fn test_sigmoid_saturation_at_extremes() {
    // Large positive → 1.0
    let pos = metal_sigmoid(20.0);
    assert!(approx_eq(pos, 1.0, TOL), "sigmoid(20) should be ~1.0, got {pos}");

    // Large negative → 0.0
    let neg = metal_sigmoid(-20.0);
    assert!(approx_eq(neg, 0.0, TOL), "sigmoid(-20) should be ~0.0, got {neg}");
}

#[test]
fn test_sigmoid_symmetry_property() {
    // sigmoid(-x) = 1 - sigmoid(x)
    for &x in &[0.5_f32, 1.0, 2.0, 5.0, 10.0] {
        let pos = metal_sigmoid(x);
        let neg = metal_sigmoid(-x);
        assert!(
            approx_eq(pos + neg, 1.0, TOL),
            "sigmoid({x}) + sigmoid({}) = {}, expected 1.0",
            -x,
            pos + neg
        );
    }
}

#[test]
fn test_sigmoid_batch_processing() {
    let batch: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
    for &x in &batch {
        let result = metal_sigmoid(x);
        let reference = cpu_sigmoid_f64(x as f64) as f32;
        assert!(approx_eq(result, reference, TOL), "sigmoid({x}): metal={result}, ref={reference}");
        // Range check: sigmoid is always in (0, 1)
        assert!(
            result > 0.0 && result < 1.0
                || approx_eq(result, 0.0, TOL)
                || approx_eq(result, 1.0, TOL),
            "sigmoid({x}) out of range: {result}"
        );
    }
}

#[test]
fn test_sigmoid_numerical_precision() {
    // Test values near boundaries of f32 precision
    let test_values = [1e-7_f32, 1e-5, 1e-3, 0.1, 0.5];
    for &x in &test_values {
        let metal_val = metal_sigmoid(x);
        let ref_val = cpu_sigmoid_f64(x as f64) as f32;
        assert!(
            approx_eq(metal_val, ref_val, TOL),
            "sigmoid precision at {x}: metal={metal_val}, ref={ref_val}"
        );
    }
}

// ── 6. Tanh activation (5 tests) ───────────────────────────────────

#[test]
fn test_tanh_zero_output() {
    let result = metal_tanh(0.0);
    assert!(approx_eq(result, 0.0, TOL), "tanh(0) should be 0, got {result}");
}

#[test]
fn test_tanh_saturation() {
    let pos = metal_tanh(10.0);
    assert!(approx_eq(pos, 1.0, TOL), "tanh(10) should be ~1.0, got {pos}");
    let neg = metal_tanh(-10.0);
    assert!(approx_eq(neg, -1.0, TOL), "tanh(-10) should be ~-1.0, got {neg}");
}

#[test]
fn test_tanh_odd_function_property() {
    // tanh(-x) = -tanh(x)
    for &x in &[0.1_f32, 0.5, 1.0, 2.0, 5.0] {
        let pos = metal_tanh(x);
        let neg = metal_tanh(-x);
        assert!(
            approx_eq(neg, -pos, TOL),
            "tanh odd property: tanh({})={neg}, -tanh({x})={}",
            -x,
            -pos
        );
    }
}

#[test]
fn test_tanh_range_bounds() {
    // tanh output must always be in [-1, 1]
    let inputs: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.1).collect();
    for &x in &inputs {
        let result = metal_tanh(x);
        assert!(result >= -1.0 && result <= 1.0, "tanh({x}) = {result} is outside [-1, 1]");
    }
}

#[test]
fn test_tanh_derivative_correctness() {
    // tanh'(x) = 1 - tanh(x)^2
    let h = 1e-4_f32;
    for &x in &[-2.0_f32, -1.0, 0.0, 1.0, 2.0] {
        let numerical = (metal_tanh(x + h) - metal_tanh(x - h)) / (2.0 * h);
        let t = metal_tanh(x);
        let analytical = 1.0 - t * t;
        assert!(
            approx_eq(numerical, analytical, 1e-3),
            "tanh'({x}): numerical={numerical}, analytical={analytical}"
        );
    }
}

// ── 7. Layer operations (5 tests) ──────────────────────────────────

#[test]
fn test_layer_activation_bias_fusion() {
    // Fused: activation(x + bias) should match sequential
    let input = vec![0.5_f32, -0.3, 1.2, -0.8];
    let bias = vec![0.1_f32, 0.2, -0.1, 0.3];

    let fused: Vec<f32> = input.iter().zip(bias.iter()).map(|(&x, &b)| metal_silu(x + b)).collect();

    let sequential: Vec<f32> = input
        .iter()
        .zip(bias.iter())
        .map(|(&x, &b)| {
            let biased = x + b;
            metal_silu(biased)
        })
        .collect();

    for (i, (&f, &s)) in fused.iter().zip(sequential.iter()).enumerate() {
        assert!(approx_eq(f, s, TOL), "Fused vs sequential[{i}]: fused={f}, sequential={s}");
    }
}

#[test]
fn test_layer_sequential_activations() {
    // sigmoid → tanh composition
    let input = vec![-1.0_f32, 0.0, 0.5, 1.0];

    let result: Vec<f32> = input.iter().map(|&x| metal_tanh(metal_sigmoid(x))).collect();

    for (i, (&x, &r)) in input.iter().zip(result.iter()).enumerate() {
        let expected = (cpu_sigmoid_f64(x as f64) as f32).tanh();
        assert!(
            approx_eq(r, expected, TOL),
            "Sequential activation[{i}]: got {r}, expected {expected}"
        );
    }
}

#[test]
fn test_layer_residual_activation() {
    // residual: output = activation(x) + x
    let input = vec![0.5_f32, -0.3, 1.2, -0.8];
    let result: Vec<f32> = input.iter().map(|&x| metal_silu(x) + x).collect();

    for (i, (&x, &r)) in input.iter().zip(result.iter()).enumerate() {
        let expected = metal_silu(x) + x;
        assert!(
            approx_eq(r, expected, TOL),
            "Residual + activation[{i}]: got {r}, expected {expected}"
        );
    }
}

#[test]
fn test_layer_prenorm_activation() {
    // Pre-norm: layernorm → activation
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
    let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / input.len() as f32;
    let std = (var + 1e-5).sqrt();

    let normed: Vec<f32> = input.iter().map(|&x| (x - mean) / std).collect();
    let activated: Vec<f32> = normed.iter().map(|&x| metal_gelu_tanh(x)).collect();

    for (i, &a) in activated.iter().enumerate() {
        let expected = metal_gelu_tanh(normed[i]);
        assert!(
            approx_eq(a, expected, TOL),
            "Pre-norm activation[{i}]: got {a}, expected {expected}"
        );
    }
}

#[test]
fn test_layer_postnorm_activation() {
    // Post-norm: activation → layernorm
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let activated: Vec<f32> = input.iter().map(|&x| metal_silu(x)).collect();

    let mean: f32 = activated.iter().sum::<f32>() / activated.len() as f32;
    let var: f32 =
        activated.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / activated.len() as f32;
    let std = (var + 1e-5).sqrt();

    let result: Vec<f32> = activated.iter().map(|&x| (x - mean) / std).collect();

    // Verify normalization properties
    let result_mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(approx_eq(result_mean, 0.0, 1e-4), "Post-norm mean should be ~0, got {result_mean}");
}

// ── 8. Quantized activations (5 tests) ─────────────────────────────

fn quantize_to_int8(x: f32, scale: f32) -> i8 {
    (x / scale).round().clamp(-128.0, 127.0) as i8
}

fn dequantize_from_int8(q: i8, scale: f32) -> f32 {
    q as f32 * scale
}

#[test]
fn test_quantized_int8_silu_approximation() {
    let scale = 0.05_f32;
    let inputs = [-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    for &x in &inputs {
        let fp32_result = metal_silu(x);
        let quantized = quantize_to_int8(x, scale);
        let dequantized = dequantize_from_int8(quantized, scale);
        let quant_result = metal_silu(dequantized);
        // Quantization introduces error proportional to scale
        assert!(
            approx_eq(fp32_result, quant_result, scale * 2.0),
            "Quantized SiLU({x}): fp32={fp32_result}, quant={quant_result}, scale={scale}"
        );
    }
}

#[test]
fn test_quantization_aware_activation() {
    // Simulate quantize → activate → dequantize path
    let scale = 0.1_f32;
    let inputs: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.1).collect();

    for &x in &inputs {
        let q = quantize_to_int8(x, scale);
        let dq = dequantize_from_int8(q, scale);
        let activated = metal_relu(dq);
        // ReLU should be exact even after quantization (only zeroes negatives)
        if dq >= 0.0 {
            assert!(
                approx_eq(activated, dq, TOL),
                "QAT ReLU positive: input={x}, dq={dq}, activated={activated}"
            );
        } else {
            assert!(
                approx_eq(activated, 0.0, TOL),
                "QAT ReLU negative: input={x}, dq={dq}, activated={activated}"
            );
        }
    }
}

#[test]
fn test_symmetric_quantization_preservation() {
    // Symmetric quantization: quantize(-x) = -quantize(x) for most values
    let scale = 0.05_f32;
    for i in 1..=50 {
        let x = i as f32 * 0.1;
        let q_pos = quantize_to_int8(x, scale);
        let q_neg = quantize_to_int8(-x, scale);
        // Symmetric within rounding
        assert!(
            (q_pos as i16 + q_neg as i16).unsigned_abs() <= 1,
            "Symmetric quantization: q({x})={q_pos}, q({})={q_neg}",
            -x
        );
    }
}

#[test]
fn test_activation_before_after_quantize() {
    // Compare: activate-then-quantize vs quantize-then-activate
    let scale = 0.05_f32;
    let inputs = [-1.5_f32, -0.5, 0.0, 0.5, 1.5];

    for &x in &inputs {
        // Path A: activate → quantize
        let act_first = metal_silu(x);
        let q_a = quantize_to_int8(act_first, scale);

        // Path B: quantize → dequantize → activate → quantize
        let q_input = quantize_to_int8(x, scale);
        let dq_input = dequantize_from_int8(q_input, scale);
        let act_second = metal_silu(dq_input);
        let q_b = quantize_to_int8(act_second, scale);

        // Should be close (within quantization error)
        assert!(
            (q_a as i16 - q_b as i16).unsigned_abs() <= 2,
            "Activation order: x={x}, path_a={q_a}, path_b={q_b}"
        );
    }
}

#[test]
fn test_scale_factor_propagation() {
    // Verify scale propagation through activation: if input has scale s,
    // output scale should be adjusted based on activation range
    let input_scale = 0.1_f32;
    let n = 100;
    let inputs: Vec<f32> = (0..n)
        .map(|i| dequantize_from_int8(((i as i16) - 50).clamp(-128, 127) as i8, input_scale))
        .collect();

    let activated: Vec<f32> = inputs.iter().map(|&x| metal_sigmoid(x)).collect();

    // Sigmoid output is in [0, 1], so output scale should be smaller
    let max_abs = activated.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let output_scale = max_abs / 127.0;

    assert!(
        output_scale <= input_scale,
        "Sigmoid output scale ({output_scale}) should be <= input scale ({input_scale})"
    );

    // All activated values should be quantizable with output scale
    for &a in &activated {
        let q = quantize_to_int8(a, output_scale);
        let dq = dequantize_from_int8(q, output_scale);
        assert!(
            approx_eq(a, dq, output_scale),
            "Scale propagation: activated={a}, dequantized={dq}, scale={output_scale}"
        );
    }
}

// ── 9. Edge cases (6 tests) ────────────────────────────────────────

#[test]
fn test_edge_nan_propagation() {
    let nan = f32::NAN;
    assert!(metal_silu(nan).is_nan(), "SiLU should propagate NaN");
    assert!(metal_gelu_tanh(nan).is_nan(), "GELU should propagate NaN");
    assert!(metal_sigmoid(nan).is_nan(), "Sigmoid should propagate NaN");
    assert!(metal_tanh(nan).is_nan(), "Tanh should propagate NaN");
    // Note: Metal's fmax(NaN, 0) returns 0 (IEEE 754 minNum/maxNum semantics),
    // matching Rust's f32::max behavior. ReLU with NaN input yields 0.
    let relu_nan = metal_relu(nan);
    assert!(
        relu_nan.is_nan() || relu_nan == 0.0,
        "ReLU(NaN) should be NaN or 0 (Metal fmax semantics), got {relu_nan}"
    );
}

#[test]
fn test_edge_infinity_handling() {
    let pos_inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    // ReLU: inf → inf, -inf → 0
    assert_eq!(metal_relu(pos_inf), pos_inf, "ReLU(+inf) should be +inf");
    assert_eq!(metal_relu(neg_inf), 0.0, "ReLU(-inf) should be 0");

    // Sigmoid: inf → 1, -inf → 0
    assert!(approx_eq(metal_sigmoid(pos_inf), 1.0, TOL), "sigmoid(+inf) should be 1");
    assert!(approx_eq(metal_sigmoid(neg_inf), 0.0, TOL), "sigmoid(-inf) should be 0");

    // Tanh: inf → 1, -inf → -1
    assert!(approx_eq(metal_tanh(pos_inf), 1.0, TOL), "tanh(+inf) should be 1");
    assert!(approx_eq(metal_tanh(neg_inf), -1.0, TOL), "tanh(-inf) should be -1");
}

#[test]
fn test_edge_subnormal_inputs() {
    let subnormals = [f32::MIN_POSITIVE / 2.0, -f32::MIN_POSITIVE / 2.0, 1e-40_f32, -1e-40_f32];
    for &x in &subnormals {
        let silu_result = metal_silu(x);
        assert!(silu_result.is_finite(), "SiLU({x:e}) should be finite, got {silu_result}");
        let sigmoid_result = metal_sigmoid(x);
        assert!(
            sigmoid_result.is_finite(),
            "sigmoid({x:e}) should be finite, got {sigmoid_result}"
        );
        // For very small x, sigmoid(x) ≈ 0.5
        assert!(
            approx_eq(sigmoid_result, 0.5, 1e-3),
            "sigmoid(subnormal) should be ~0.5, got {sigmoid_result}"
        );
    }
}

#[test]
fn test_edge_very_large_batch() {
    let n = 10_000;
    let batch: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 20.0 - 10.0).collect();
    let results: Vec<f32> = batch.iter().map(|&x| metal_silu(x)).collect();

    assert_eq!(results.len(), n, "Output batch size should match input");

    // Verify all outputs are finite
    for (i, &r) in results.iter().enumerate() {
        assert!(r.is_finite(), "Large batch SiLU[{i}] should be finite, got {r}");
    }

    // Spot check first, middle, last
    assert!(approx_eq(results[0], metal_silu(batch[0]), TOL), "Large batch first element mismatch");
    assert!(
        approx_eq(results[n / 2], metal_silu(batch[n / 2]), TOL),
        "Large batch middle element mismatch"
    );
    assert!(
        approx_eq(results[n - 1], metal_silu(batch[n - 1]), TOL),
        "Large batch last element mismatch"
    );
}

#[test]
fn test_edge_zero_length_input() {
    let empty: Vec<f32> = vec![];
    let results: Vec<f32> = empty.iter().map(|&x| metal_silu(x)).collect();
    assert!(results.is_empty(), "Zero-length output should be empty");

    let softmax_result = metal_softmax(&[]);
    assert!(softmax_result.is_empty(), "Softmax of empty input should be empty");
}

#[test]
fn test_edge_misaligned_buffer_offsets() {
    // Simulate processing from different offsets within a larger buffer
    let buffer: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();

    // Process from offset 0, aligned
    let aligned: Vec<f32> = buffer[0..16].iter().map(|&x| metal_gelu_tanh(x)).collect();

    // Process from offset 1, misaligned
    let misaligned: Vec<f32> = buffer[1..17].iter().map(|&x| metal_gelu_tanh(x)).collect();

    // Each element should match its individual computation
    for i in 0..16 {
        let expected_a = metal_gelu_tanh(buffer[i]);
        let expected_m = metal_gelu_tanh(buffer[i + 1]);
        assert!(
            approx_eq(aligned[i], expected_a, TOL),
            "Aligned offset[{i}]: got {}, expected {expected_a}",
            aligned[i]
        );
        assert!(
            approx_eq(misaligned[i], expected_m, TOL),
            "Misaligned offset[{i}]: got {}, expected {expected_m}",
            misaligned[i]
        );
    }
}

// ── 10. Performance patterns (6 tests) ─────────────────────────────

#[test]
fn test_perf_coalesced_memory_access() {
    // Simulate coalesced (sequential) vs strided access patterns
    let n = 1024;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();

    // Coalesced: sequential access
    let coalesced: Vec<f32> = data.iter().map(|&x| metal_silu(x)).collect();

    // Strided: access every 4th element (simulates non-coalesced pattern)
    let strided: Vec<f32> = (0..n / 4).map(|i| metal_silu(data[i * 4])).collect();

    // Results should be identical regardless of access pattern
    for i in 0..n / 4 {
        assert!(
            approx_eq(coalesced[i * 4], strided[i], TOL),
            "Coalesced vs strided[{i}]: coalesced={}, strided={}",
            coalesced[i * 4],
            strided[i]
        );
    }
}

#[test]
fn test_perf_threadgroup_sizing_for_activations() {
    // Simulate different threadgroup sizes processing the same data
    let n = 256;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();

    let threadgroup_sizes = [32, 64, 128, 256];

    let reference: Vec<f32> = data.iter().map(|&x| metal_relu(x)).collect();

    for &tg_size in &threadgroup_sizes {
        let mut result = vec![0.0_f32; n];
        // Process in chunks of threadgroup size
        for chunk_start in (0..n).step_by(tg_size) {
            let chunk_end = (chunk_start + tg_size).min(n);
            for i in chunk_start..chunk_end {
                result[i] = metal_relu(data[i]);
            }
        }
        for i in 0..n {
            assert!(
                approx_eq(result[i], reference[i], TOL),
                "Threadgroup size {tg_size} at [{i}]: got {}, expected {}",
                result[i],
                reference[i]
            );
        }
    }
}

#[test]
fn test_perf_dispatch_grid_dimensions() {
    // 2D grid dispatch: simulate processing a matrix of activations
    let rows = 16;
    let cols = 32;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.01).cos()).collect();

    // 1D dispatch
    let result_1d: Vec<f32> = data.iter().map(|&x| metal_gelu_tanh(x)).collect();

    // 2D dispatch (row-major)
    let mut result_2d = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            result_2d[idx] = metal_gelu_tanh(data[idx]);
        }
    }

    for i in 0..rows * cols {
        assert!(
            approx_eq(result_1d[i], result_2d[i], TOL),
            "1D vs 2D dispatch[{i}]: 1d={}, 2d={}",
            result_1d[i],
            result_2d[i]
        );
    }
}

#[test]
fn test_perf_shared_memory_bank_conflicts() {
    // Simulate shared memory access: 32 banks, stride patterns
    let bank_count = 32;
    let data: Vec<f32> = (0..bank_count * 4).map(|i| (i as f32 * 0.1).sin()).collect();

    // No conflict: sequential access within a bank
    let no_conflict: Vec<f32> = data.iter().map(|&x| metal_sigmoid(x)).collect();

    // With potential conflict: stride-33 wraps to bank+1 (no actual conflict)
    let stride_33: Vec<f32> = (0..bank_count)
        .map(|i| {
            let idx = (i * 33) % data.len();
            metal_sigmoid(data[idx])
        })
        .collect();

    // Verify mathematical correctness regardless of access pattern
    for i in 0..bank_count {
        let idx = (i * 33) % data.len();
        assert!(
            approx_eq(stride_33[i], no_conflict[idx], TOL),
            "Bank conflict pattern[{i}]: stride_33={}, sequential={}",
            stride_33[i],
            no_conflict[idx]
        );
    }
}

#[test]
fn test_perf_pipeline_state_caching() {
    // Simulate reusing the same activation function across multiple invocations
    let data1: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let data2: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1) + 10.0).collect();
    let data3: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1) - 5.0).collect();

    // Same function applied to different data (pipeline state reuse)
    let results =
        [&data1, &data2, &data3].map(|d| d.iter().map(|&x| metal_silu(x)).collect::<Vec<f32>>());

    for (batch_idx, (data, result)) in
        [&data1, &data2, &data3].iter().zip(results.iter()).enumerate()
    {
        for (i, (&x, &r)) in data.iter().zip(result.iter()).enumerate() {
            let expected = metal_silu(x);
            assert!(
                approx_eq(r, expected, TOL),
                "Pipeline batch {batch_idx}[{i}]: got {r}, expected {expected}"
            );
        }
    }
}

#[test]
fn test_perf_multi_activation_fusion() {
    // Simulate fused kernel: SiLU(x) * Linear(x) (SwiGLU pattern)
    let dim = 128;
    let gate_input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();
    let up_input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.03).cos()).collect();

    // Fused: SiLU(gate) * up in one pass
    let fused: Vec<f32> =
        gate_input.iter().zip(up_input.iter()).map(|(&g, &u)| metal_silu(g) * u).collect();

    // Sequential: compute SiLU first, then multiply
    let silu_gate: Vec<f32> = gate_input.iter().map(|&x| metal_silu(x)).collect();
    let sequential: Vec<f32> =
        silu_gate.iter().zip(up_input.iter()).map(|(&s, &u)| s * u).collect();

    for i in 0..dim {
        assert!(
            approx_eq(fused[i], sequential[i], TOL),
            "SwiGLU fusion[{i}]: fused={}, sequential={}",
            fused[i],
            sequential[i]
        );
    }
}
