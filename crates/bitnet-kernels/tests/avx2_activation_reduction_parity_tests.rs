#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for activation functions and reduction operations.
//!
//! Each test exercises a kernel that has an explicit AVX2 fast-path
//! (via `is_x86_feature_detected!("avx2")` runtime dispatch) and
//! compares the output against a pure-scalar reference within defined
//! tolerances.
//!
//! Test dimensions are chosen to exercise both the 8-wide SIMD body
//! and the scalar tail (lengths not divisible by 8).
//!
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::simd_activation_functions::{
    simd_gelu, simd_gelu_inplace, simd_sigmoid, simd_silu, simd_silu_inplace,
};
use bitnet_kernels::cpu::simd_reduction::{
    simd_argmax, simd_argmin, simd_horizontal_max, simd_horizontal_min, simd_horizontal_sum,
};

#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_parity::{assert_vec_parity, close, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────
//
// Activation functions use approximated exp/tanh internally so we allow
// slightly wider tolerances than softmax/layer-norm.

/// Absolute tolerance for activation functions (gelu, silu, sigmoid).
const ELEM_ABS_TOL: f32 = 1e-5;
/// Relative tolerance for activation functions.
const ELEM_REL_TOL: f32 = 1e-4;

/// Absolute tolerance for reduction operations (sum, max, min).
const REDUCTION_ABS_TOL: f32 = 1e-5;
/// Relative tolerance for reduction operations.
const REDUCTION_REL_TOL: f32 = 1e-5;

// ── Scalar reference implementations ──────────────────────────────────

fn reference_gelu(x: f32) -> f32 {
    // simd_gelu uses the fast sigmoid approximation: x * sigmoid(1.702 * x)
    x * reference_sigmoid(1.702 * x)
}

fn reference_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn reference_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Activation sizes ──────────────────────────────────────────────────

const ACTIVATION_SIZES: &[usize] = &[1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 256, 1024];
const REDUCTION_SIZES: &[usize] = &[1, 7, 8, 9, 16, 31, 32, 33, 64, 128, 256, 1024];

// ════════════════════════════════════════════════════════════════════════
// 1. GELU — simd_gelu dispatches to AVX2 or scalar internally
// ════════════════════════════════════════════════════════════════════════

/// GELU parity: dispatch path vs pure-scalar reference.
#[test]
fn gelu_parity() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 100 + n as u64);
        let expected: Vec<f32> = input.iter().map(|&x| reference_gelu(x)).collect();
        let mut actual = vec![0.0f32; n];
        simd_gelu(&input, &mut actual).expect("gelu should not fail");
        assert_vec_parity(&actual, &expected, ELEM_ABS_TOL, ELEM_REL_TOL, &format!("gelu(n={n})"));
    }
}

/// GELU inplace parity: dispatch path vs pure-scalar reference.
#[test]
fn gelu_inplace_parity() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 200 + n as u64);
        let expected: Vec<f32> = input.iter().map(|&x| reference_gelu(x)).collect();
        let mut actual = input.clone();
        simd_gelu_inplace(&mut actual).expect("gelu_inplace should not fail");
        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("gelu_inplace(n={n})"),
        );
    }
}

/// GELU: out-of-place and inplace must produce identical results.
#[test]
fn gelu_outofplace_vs_inplace() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 300 + n as u64);
        let mut out = vec![0.0f32; n];
        simd_gelu(&input, &mut out).expect("gelu should not fail");
        let mut inplace = input.clone();
        simd_gelu_inplace(&mut inplace).expect("gelu_inplace should not fail");
        assert_vec_parity(&out, &inplace, 0.0, 0.0, &format!("gelu_outofplace_vs_inplace(n={n})"));
    }
}

// ════════════════════════════════════════════════════════════════════════
// 2. SiLU — simd_silu dispatches to AVX2 or scalar internally
// ════════════════════════════════════════════════════════════════════════

/// SiLU parity: dispatch path vs pure-scalar reference.
#[test]
fn silu_parity() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 400 + n as u64);
        let expected: Vec<f32> = input.iter().map(|&x| reference_silu(x)).collect();
        let mut actual = vec![0.0f32; n];
        simd_silu(&input, &mut actual).expect("silu should not fail");
        assert_vec_parity(&actual, &expected, ELEM_ABS_TOL, ELEM_REL_TOL, &format!("silu(n={n})"));
    }
}

/// SiLU inplace parity: dispatch path vs pure-scalar reference.
#[test]
fn silu_inplace_parity() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 500 + n as u64);
        let expected: Vec<f32> = input.iter().map(|&x| reference_silu(x)).collect();
        let mut actual = input.clone();
        simd_silu_inplace(&mut actual).expect("silu_inplace should not fail");
        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("silu_inplace(n={n})"),
        );
    }
}

/// SiLU: out-of-place and inplace must produce identical results.
#[test]
fn silu_outofplace_vs_inplace() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 600 + n as u64);
        let mut out = vec![0.0f32; n];
        simd_silu(&input, &mut out).expect("silu should not fail");
        let mut inplace = input.clone();
        simd_silu_inplace(&mut inplace).expect("silu_inplace should not fail");
        assert_vec_parity(&out, &inplace, 0.0, 0.0, &format!("silu_outofplace_vs_inplace(n={n})"));
    }
}

// ════════════════════════════════════════════════════════════════════════
// 3. SIGMOID — simd_sigmoid dispatches to AVX2 or scalar internally
// ════════════════════════════════════════════════════════════════════════

/// Sigmoid parity: dispatch path vs pure-scalar reference.
#[test]
fn sigmoid_parity() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 700 + n as u64);
        let expected: Vec<f32> = input.iter().map(|&x| reference_sigmoid(x)).collect();
        let mut actual = vec![0.0f32; n];
        simd_sigmoid(&input, &mut actual).expect("sigmoid should not fail");
        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("sigmoid(n={n})"),
        );
    }
}

/// Sigmoid outputs must always be in [0, 1].
#[test]
fn sigmoid_output_range() {
    for &n in ACTIVATION_SIZES {
        let input = pseudo_rand(n, 750 + n as u64);
        let mut output = vec![0.0f32; n];
        simd_sigmoid(&input, &mut output).expect("sigmoid should not fail");
        for (i, &v) in output.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "sigmoid(n={n})[{i}] = {v} is outside [0, 1]");
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// 4. ACTIVATION NUMERICAL STABILITY
// ════════════════════════════════════════════════════════════════════════

/// Activations must produce finite outputs for large-magnitude inputs.
#[test]
fn activation_stability_large_values() {
    let input = vec![88.0, -88.0, 0.0, 50.0, -50.0, 1e-8, -1e-8, 42.0];
    let n = input.len();

    let mut gelu_out = vec![0.0f32; n];
    simd_gelu(&input, &mut gelu_out).expect("gelu should not fail");
    for (i, &v) in gelu_out.iter().enumerate() {
        assert!(v.is_finite(), "gelu output[{i}] is not finite: {v}");
    }

    let mut silu_out = vec![0.0f32; n];
    simd_silu(&input, &mut silu_out).expect("silu should not fail");
    for (i, &v) in silu_out.iter().enumerate() {
        assert!(v.is_finite(), "silu output[{i}] is not finite: {v}");
    }

    let mut sigmoid_out = vec![0.0f32; n];
    simd_sigmoid(&input, &mut sigmoid_out).expect("sigmoid should not fail");
    for (i, &v) in sigmoid_out.iter().enumerate() {
        assert!(v.is_finite(), "sigmoid output[{i}] is not finite: {v}");
        assert!((0.0..=1.0).contains(&v), "sigmoid output[{i}] = {v} is outside [0, 1]");
    }
}

// ════════════════════════════════════════════════════════════════════════
// 5. HORIZONTAL SUM — simd_horizontal_sum vs scalar sum
// ════════════════════════════════════════════════════════════════════════

/// Horizontal sum parity: dispatch path vs scalar `iter().sum()`.
#[test]
fn horizontal_sum_parity() {
    for &n in REDUCTION_SIZES {
        let data = pseudo_rand(n, 1000 + n as u64);
        let expected: f32 = data.iter().sum();
        let actual = simd_horizontal_sum(&data).expect("horizontal_sum should not fail");
        assert!(
            close(actual, expected, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
            "horizontal_sum(n={n}): scalar={expected}, dispatched={actual} (diff={})",
            (actual - expected).abs()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 6. HORIZONTAL MAX — simd_horizontal_max vs scalar max
// ════════════════════════════════════════════════════════════════════════

/// Horizontal max parity: dispatch path vs scalar fold.
#[test]
fn horizontal_max_parity() {
    for &n in REDUCTION_SIZES {
        let data = pseudo_rand(n, 2000 + n as u64);
        let expected = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let actual = simd_horizontal_max(&data).expect("horizontal_max should not fail");
        assert!(
            close(actual, expected, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
            "horizontal_max(n={n}): scalar={expected}, dispatched={actual} (diff={})",
            (actual - expected).abs()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 7. HORIZONTAL MIN — simd_horizontal_min vs scalar min
// ════════════════════════════════════════════════════════════════════════

/// Horizontal min parity: dispatch path vs scalar fold.
#[test]
fn horizontal_min_parity() {
    for &n in REDUCTION_SIZES {
        let data = pseudo_rand(n, 3000 + n as u64);
        let expected = data.iter().copied().fold(f32::INFINITY, f32::min);
        let actual = simd_horizontal_min(&data).expect("horizontal_min should not fail");
        assert!(
            close(actual, expected, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
            "horizontal_min(n={n}): scalar={expected}, dispatched={actual} (diff={})",
            (actual - expected).abs()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 8. ARGMAX — simd_argmax vs scalar argmax
// ════════════════════════════════════════════════════════════════════════

/// Argmax parity: dispatch path vs scalar argmax.
#[test]
fn argmax_parity() {
    for &n in REDUCTION_SIZES {
        let data = pseudo_rand(n, 4000 + n as u64);

        // Scalar reference argmax
        let mut expected_idx = 0;
        let mut expected_val = f32::NEG_INFINITY;
        for i in 0..data.len() {
            if data[i] > expected_val {
                expected_val = data[i];
                expected_idx = i;
            }
        }

        let result = simd_argmax(&data).expect("argmax should not fail");
        assert_eq!(
            result.index, expected_idx,
            "argmax(n={n}): expected index={expected_idx}, got={}",
            result.index
        );
        assert!(
            close(result.value, expected_val, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
            "argmax(n={n}): expected value={expected_val}, got={} (diff={})",
            result.value,
            (result.value - expected_val).abs()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 9. ARGMIN — simd_argmin vs scalar argmin
// ════════════════════════════════════════════════════════════════════════

/// Argmin parity: dispatch path vs scalar argmin.
#[test]
fn argmin_parity() {
    for &n in REDUCTION_SIZES {
        let data = pseudo_rand(n, 5000 + n as u64);

        // Scalar reference argmin
        let mut expected_idx = 0;
        let mut expected_val = f32::INFINITY;
        for i in 0..data.len() {
            if data[i] < expected_val {
                expected_val = data[i];
                expected_idx = i;
            }
        }

        let result = simd_argmin(&data).expect("argmin should not fail");
        assert_eq!(
            result.index, expected_idx,
            "argmin(n={n}): expected index={expected_idx}, got={}",
            result.index
        );
        assert!(
            close(result.value, expected_val, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
            "argmin(n={n}): expected value={expected_val}, got={} (diff={})",
            result.value,
            (result.value - expected_val).abs()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 10. REDUCTION EDGE CASES
// ════════════════════════════════════════════════════════════════════════

/// Single-element reductions must return that element.
#[test]
fn reduction_single_element() {
    let data = vec![42.0f32];

    let sum = simd_horizontal_sum(&data).expect("sum single");
    assert!((sum - 42.0).abs() < 1e-6, "sum of [42.0] = {sum}");

    let max = simd_horizontal_max(&data).expect("max single");
    assert!((max - 42.0).abs() < 1e-6, "max of [42.0] = {max}");

    let min = simd_horizontal_min(&data).expect("min single");
    assert!((min - 42.0).abs() < 1e-6, "min of [42.0] = {min}");

    let amax = simd_argmax(&data).expect("argmax single");
    assert_eq!(amax.index, 0);
    assert!((amax.value - 42.0).abs() < 1e-6);

    let amin = simd_argmin(&data).expect("argmin single");
    assert_eq!(amin.index, 0);
    assert!((amin.value - 42.0).abs() < 1e-6);
}

/// Constant-valued input: all elements equal.
#[test]
fn reduction_constant_input() {
    let data = vec![std::f32::consts::PI; 33];

    let sum = simd_horizontal_sum(&data).expect("sum const");
    let expected_sum = std::f32::consts::PI * 33.0;
    assert!(
        close(sum, expected_sum, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
        "sum of 33 × PI: expected={expected_sum}, got={sum}"
    );

    let max = simd_horizontal_max(&data).expect("max const");
    assert!(
        close(max, std::f32::consts::PI, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
        "max of constant PI: got={max}"
    );

    let min = simd_horizontal_min(&data).expect("min const");
    assert!(
        close(min, std::f32::consts::PI, REDUCTION_ABS_TOL, REDUCTION_REL_TOL),
        "min of constant PI: got={min}"
    );

    // argmax/argmin should return index 0 for constant input (first occurrence)
    let amax = simd_argmax(&data).expect("argmax const");
    assert_eq!(amax.index, 0, "argmax of constant should return first index");

    let amin = simd_argmin(&data).expect("argmin const");
    assert_eq!(amin.index, 0, "argmin of constant should return first index");
}
