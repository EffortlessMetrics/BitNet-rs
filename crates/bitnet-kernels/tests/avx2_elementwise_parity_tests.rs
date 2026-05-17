#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for element-wise operations.
//!
//! Each test exercises a function from `bitnet_kernels::cpu::elementwise_ops`
//! that dispatches to AVX2 intrinsics on x86-64 (8-wide SIMD body) with
//! scalar fallback for tails and other architectures, then compares the
//! output against a pure-scalar reference within defined tolerances.
//!
//! Test dimensions:
//!   - 63: not divisible by 8, exercises scalar tail handling
//!   - 128: 8-aligned, exercises pure SIMD body
//!
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::elementwise_ops;
use std::f32::consts::PI;

#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_parity::{assert_vec_parity, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────
//
// Arithmetic ops (add/sub/mul/div/fma) should be bit-exact or within
// a single ULP; transcendental and activation functions may diverge
// more due to polynomial approximations in AVX2 fast-paths.

/// Absolute tolerance for arithmetic kernels (add, sub, mul, div, fma).
const ARITH_ABS_TOL: f32 = 1e-6;
/// Relative tolerance for arithmetic kernels.
const ARITH_REL_TOL: f32 = 1e-6;

/// Absolute tolerance for transcendental / activation kernels.
const TRANS_ABS_TOL: f32 = 1e-5;
/// Relative tolerance for transcendental / activation kernels.
const TRANS_REL_TOL: f32 = 1e-4;

/// Test lengths: 63 (non-8-aligned, exercises scalar tail) and 128 (8-aligned).
const LENGTHS: [usize; 2] = [63, 128];

// ════════════════════════════════════════════════════════════════════════
// 1. ARITHMETIC OPS
// ════════════════════════════════════════════════════════════════════════

/// Add parity: dispatched path vs `a[i] + b[i]`.
#[test]
fn test_add_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 100 + n as u64);
        let b = pseudo_rand(n, 200 + n as u64);
        let actual = elementwise_ops::add(&a, &b);
        let expected: Vec<f32> = (0..n).map(|i| a[i] + b[i]).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("add(n={n})"));
    }
}

/// Sub parity: dispatched path vs `a[i] - b[i]`.
#[test]
fn test_sub_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 300 + n as u64);
        let b = pseudo_rand(n, 400 + n as u64);
        let actual = elementwise_ops::sub(&a, &b);
        let expected: Vec<f32> = (0..n).map(|i| a[i] - b[i]).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("sub(n={n})"));
    }
}

/// Mul parity: dispatched path vs `a[i] * b[i]`.
#[test]
fn test_mul_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 500 + n as u64);
        let b = pseudo_rand(n, 600 + n as u64);
        let actual = elementwise_ops::mul(&a, &b);
        let expected: Vec<f32> = (0..n).map(|i| a[i] * b[i]).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("mul(n={n})"));
    }
}

/// Div parity: dispatched path vs `a[i] / b[i]`.
/// Values of `b` are shifted to [0.5, 1.5] to avoid division by near-zero.
#[test]
fn test_div_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 700 + n as u64);
        let b_raw = pseudo_rand(n, 800 + n as u64);
        let b: Vec<f32> = b_raw.iter().map(|&v| v.abs() + 0.5).collect();
        let actual = elementwise_ops::div(&a, &b);
        let expected: Vec<f32> = (0..n).map(|i| a[i] / b[i]).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("div(n={n})"));
    }
}

// ════════════════════════════════════════════════════════════════════════
// 2. FMA
// ════════════════════════════════════════════════════════════════════════

/// Fused multiply-add parity: dispatched path vs `a[i]*b[i]+c[i]`.
#[test]
fn test_fused_multiply_add_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 900 + n as u64);
        let b = pseudo_rand(n, 1000 + n as u64);
        let c = pseudo_rand(n, 1100 + n as u64);
        let actual = elementwise_ops::fused_multiply_add(&a, &b, &c);
        let expected: Vec<f32> = (0..n).map(|i| a[i] * b[i] + c[i]).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("fma(n={n})"));
    }
}

// ════════════════════════════════════════════════════════════════════════
// 3. TRANSCENDENTAL
// ════════════════════════════════════════════════════════════════════════

/// Exp parity: dispatched path vs `a[i].exp()`.
/// Input is scaled to [-2, 2] to avoid overflow.
#[test]
fn test_exp_parity() {
    for &n in &LENGTHS {
        let raw = pseudo_rand(n, 1200 + n as u64);
        let a: Vec<f32> = raw.iter().map(|&v| v * 2.0).collect();
        let actual = elementwise_ops::exp(&a);
        let expected: Vec<f32> = (0..n).map(|i| a[i].exp()).collect();
        assert_vec_parity(&actual, &expected, TRANS_ABS_TOL, TRANS_REL_TOL, &format!("exp(n={n})"));
    }
}

/// Sqrt parity: dispatched path vs `a[i].sqrt()`.
/// Input values are made positive via `abs()`.
#[test]
fn test_sqrt_parity() {
    for &n in &LENGTHS {
        let raw = pseudo_rand(n, 1300 + n as u64);
        let a: Vec<f32> = raw.iter().map(|&v| v.abs()).collect();
        let actual = elementwise_ops::sqrt(&a);
        let expected: Vec<f32> = (0..n).map(|i| a[i].sqrt()).collect();
        assert_vec_parity(
            &actual,
            &expected,
            TRANS_ABS_TOL,
            TRANS_REL_TOL,
            &format!("sqrt(n={n})"),
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 4. ACTIVATION FUNCTIONS
// ════════════════════════════════════════════════════════════════════════

/// Sigmoid parity: dispatched path vs `1.0 / (1.0 + (-a[i]).exp())`.
#[test]
fn test_sigmoid_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 1400 + n as u64);
        let actual = elementwise_ops::sigmoid(&a);
        let expected: Vec<f32> = (0..n).map(|i| 1.0 / (1.0 + (-a[i]).exp())).collect();
        assert_vec_parity(
            &actual,
            &expected,
            TRANS_ABS_TOL,
            TRANS_REL_TOL,
            &format!("sigmoid(n={n})"),
        );
    }
}

/// GELU parity: dispatched path vs scalar GELU formula
/// `0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
#[test]
fn test_gelu_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 1500 + n as u64);
        let actual = elementwise_ops::gelu(&a);
        let expected: Vec<f32> = (0..n)
            .map(|i| {
                let x = a[i];
                0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044_715 * x.powi(3))).tanh())
            })
            .collect();
        assert_vec_parity(
            &actual,
            &expected,
            TRANS_ABS_TOL,
            TRANS_REL_TOL,
            &format!("gelu(n={n})"),
        );
    }
}

/// SiLU parity: dispatched path vs `a[i] * (1.0 / (1.0 + (-a[i]).exp()))`.
#[test]
fn test_silu_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 1600 + n as u64);
        let actual = elementwise_ops::silu(&a);
        let expected: Vec<f32> = (0..n).map(|i| a[i] * (1.0 / (1.0 + (-a[i]).exp()))).collect();
        assert_vec_parity(
            &actual,
            &expected,
            TRANS_ABS_TOL,
            TRANS_REL_TOL,
            &format!("silu(n={n})"),
        );
    }
}

/// ReLU parity: dispatched path vs `a[i].max(0.0)`.
#[test]
fn test_relu_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 1700 + n as u64);
        let actual = elementwise_ops::relu(&a);
        let expected: Vec<f32> = (0..n).map(|i| a[i].max(0.0)).collect();
        assert_vec_parity(
            &actual,
            &expected,
            ARITH_ABS_TOL,
            ARITH_REL_TOL,
            &format!("relu(n={n})"),
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 5. MISC
// ════════════════════════════════════════════════════════════════════════

/// Abs parity: dispatched path vs `a[i].abs()`.
#[test]
fn test_abs_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 1800 + n as u64);
        let actual = elementwise_ops::abs(&a);
        let expected: Vec<f32> = (0..n).map(|i| a[i].abs()).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("abs(n={n})"));
    }
}

/// Neg parity: dispatched path vs `-a[i]`.
#[test]
fn test_neg_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 1900 + n as u64);
        let actual = elementwise_ops::neg(&a);
        let expected: Vec<f32> = (0..n).map(|i| -a[i]).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("neg(n={n})"));
    }
}

/// Clamp parity: dispatched path vs `a[i].clamp(-0.5, 0.5)`.
#[test]
fn test_clamp_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 2000 + n as u64);
        let actual = elementwise_ops::clamp(&a, -0.5, 0.5);
        let expected: Vec<f32> = (0..n).map(|i| a[i].clamp(-0.5, 0.5)).collect();
        assert_vec_parity(
            &actual,
            &expected,
            ARITH_ABS_TOL,
            ARITH_REL_TOL,
            &format!("clamp(n={n})"),
        );
    }
}

/// Min parity: dispatched path vs `a[i].min(b[i])`.
#[test]
fn test_min_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 2100 + n as u64);
        let b = pseudo_rand(n, 2200 + n as u64);
        let actual = elementwise_ops::min(&a, &b);
        let expected: Vec<f32> = (0..n).map(|i| a[i].min(b[i])).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("min(n={n})"));
    }
}

/// Max parity: dispatched path vs `a[i].max(b[i])`.
#[test]
fn test_max_parity() {
    for &n in &LENGTHS {
        let a = pseudo_rand(n, 2300 + n as u64);
        let b = pseudo_rand(n, 2400 + n as u64);
        let actual = elementwise_ops::max(&a, &b);
        let expected: Vec<f32> = (0..n).map(|i| a[i].max(b[i])).collect();
        assert_vec_parity(&actual, &expected, ARITH_ABS_TOL, ARITH_REL_TOL, &format!("max(n={n})"));
    }
}
