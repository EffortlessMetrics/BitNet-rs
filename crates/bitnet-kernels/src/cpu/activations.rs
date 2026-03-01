//! CPU activation function kernels.
//!
//! Comprehensive set of element-wise activation functions for CPU
//! inference, complementing the CUDA activations in
//! `crate::cuda::activations`.
//!
//! # Provided activations
//!
//! | Function | Formula |
//! |---|---|
//! | ReLU | `max(0, x)` |
//! | LeakyReLU | `x if x≥0, α·x otherwise` |
//! | PReLU | per-channel LeakyReLU |
//! | GELU (exact) | `0.5·x·(1 + erf(x/√2))` |
//! | GELU (fast) | tanh approximation |
//! | SiLU / Swish | `x·σ(x)` |
//! | Sigmoid | `1/(1+exp(−x))` |
//! | Tanh | `tanh(x)` |
//! | Mish | `x·tanh(softplus(x))` |
//! | HardSwish | piecewise linear approx of Swish |
//! | HardSigmoid | piecewise linear approx of Sigmoid |
//!
//! Each function has an out-of-place variant (`*_activate`) and an
//! in-place variant (`*_activate_inplace`).

use std::f32::consts::SQRT_2;

// -----------------------------------------------------------------------
// Scalar helpers
// -----------------------------------------------------------------------

/// Abramowitz & Stegun approximation of `erf(x)` (max error ≈ 1.5e-7).
#[inline]
fn erff_approx(x: f32) -> f32 {
    let sign = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly = 0.254_829_6 * t - 0.284_496_74 * t2 + 1.421_413_8 * t3 - 1.453_152_1 * t4
        + 1.061_405_4 * t5;
    sign * (1.0 - poly * (-a * a).exp())
}

/// Scalar sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scalar SiLU (Swish): `x · σ(x)`.
#[inline]
fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_scalar(x)
}

/// Scalar exact GELU: `0.5 · x · (1 + erf(x / √2))`.
#[inline]
fn gelu_exact_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + erff_approx(x / SQRT_2))
}

/// Scalar fast GELU (tanh approximation).
#[inline]
fn gelu_fast_scalar(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Scalar Mish: `x · tanh(softplus(x))` where `softplus(x) = ln(1+exp(x))`.
#[inline]
fn mish_scalar(x: f32) -> f32 {
    let sp = (1.0_f32 + x.exp()).ln();
    x * sp.tanh()
}

/// Scalar HardSwish (MobileNetV3).
#[inline]
fn hard_swish_scalar(x: f32) -> f32 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

/// Scalar HardSigmoid.
#[inline]
fn hard_sigmoid_scalar(x: f32) -> f32 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (x + 3.0) / 6.0
    }
}

// -----------------------------------------------------------------------
// Out-of-place public API
// -----------------------------------------------------------------------

/// Apply ReLU: `max(0, x)`.
pub fn relu_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.max(0.0);
    }
}

/// Apply Leaky ReLU with negative slope `alpha`.
pub fn leaky_relu_activate(input: &[f32], output: &mut [f32], alpha: f32) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = if x >= 0.0 { x } else { alpha * x };
    }
}

/// Apply PReLU with per-channel slopes.
///
/// `slopes` length must equal the channel count.  Elements in `input`
/// are assumed to be laid out as `[..., channels]` with the channel
/// dimension last.
pub fn prelu_activate(input: &[f32], output: &mut [f32], slopes: &[f32]) {
    debug_assert!(output.len() >= input.len());
    debug_assert!(!slopes.is_empty());
    let c = slopes.len();
    for (i, (&x, o)) in input.iter().zip(output.iter_mut()).enumerate() {
        let alpha = slopes[i % c];
        *o = if x >= 0.0 { x } else { alpha * x };
    }
}

/// Apply exact GELU: `0.5·x·(1 + erf(x/√2))`.
pub fn gelu_exact_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = gelu_exact_scalar(x);
    }
}

/// Apply fast GELU (tanh approximation).
pub fn gelu_fast_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = gelu_fast_scalar(x);
    }
}

/// Apply SiLU / Swish: `x·σ(x)`.
pub fn silu_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = silu_scalar(x);
    }
}

/// Apply Sigmoid: `1/(1+exp(−x))`.
pub fn sigmoid_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = sigmoid_scalar(x);
    }
}

/// Apply Tanh.
pub fn tanh_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.tanh();
    }
}

/// Apply Mish: `x·tanh(softplus(x))`.
pub fn mish_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = mish_scalar(x);
    }
}

/// Apply HardSwish (MobileNetV3).
pub fn hard_swish_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = hard_swish_scalar(x);
    }
}

/// Apply HardSigmoid.
pub fn hard_sigmoid_activate(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = hard_sigmoid_scalar(x);
    }
}

// -----------------------------------------------------------------------
// In-place public API
// -----------------------------------------------------------------------

/// In-place ReLU.
pub fn relu_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = x.max(0.0);
    }
}

/// In-place Leaky ReLU.
pub fn leaky_relu_activate_inplace(data: &mut [f32], alpha: f32) {
    for x in data.iter_mut() {
        if *x < 0.0 {
            *x *= alpha;
        }
    }
}

/// In-place exact GELU.
pub fn gelu_exact_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = gelu_exact_scalar(*x);
    }
}

/// In-place fast GELU.
pub fn gelu_fast_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = gelu_fast_scalar(*x);
    }
}

/// In-place SiLU / Swish.
pub fn silu_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = silu_scalar(*x);
    }
}

/// In-place Sigmoid.
pub fn sigmoid_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = sigmoid_scalar(*x);
    }
}

/// In-place Tanh.
pub fn tanh_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = x.tanh();
    }
}

/// In-place Mish.
pub fn mish_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = mish_scalar(*x);
    }
}

/// In-place HardSwish.
pub fn hard_swish_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = hard_swish_scalar(*x);
    }
}

/// In-place HardSigmoid.
pub fn hard_sigmoid_activate_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = hard_sigmoid_scalar(*x);
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (diff={})", (a - b).abs());
    }

    // -- ReLU --

    #[test]
    fn test_relu_positive() {
        let input = [1.0, 2.5, 0.1];
        let mut out = [0.0; 3];
        relu_activate(&input, &mut out);
        assert_eq!(out, [1.0, 2.5, 0.1]);
    }

    #[test]
    fn test_relu_negative() {
        let input = [-1.0, -0.5, 0.0];
        let mut out = [0.0; 3];
        relu_activate(&input, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_inplace() {
        let mut data = [-2.0, 0.0, 3.0];
        relu_activate_inplace(&mut data);
        assert_eq!(data, [0.0, 0.0, 3.0]);
    }

    // -- Leaky ReLU --

    #[test]
    fn test_leaky_relu() {
        let input = [1.0, -1.0, 0.0];
        let mut out = [0.0; 3];
        leaky_relu_activate(&input, &mut out, 0.01);
        assert_close(out[0], 1.0, EPS);
        assert_close(out[1], -0.01, EPS);
        assert_close(out[2], 0.0, EPS);
    }

    #[test]
    fn test_leaky_relu_inplace() {
        let mut data = [-10.0, 5.0];
        leaky_relu_activate_inplace(&mut data, 0.2);
        assert_close(data[0], -2.0, EPS);
        assert_close(data[1], 5.0, EPS);
    }

    // -- PReLU --

    #[test]
    fn test_prelu_per_channel() {
        // 2 channels, 4 elements total
        let input = [1.0, -1.0, 2.0, -2.0];
        let slopes = [0.1, 0.25];
        let mut out = [0.0; 4];
        prelu_activate(&input, &mut out, &slopes);
        assert_close(out[0], 1.0, EPS);
        assert_close(out[1], -0.25, EPS);
        assert_close(out[2], 2.0, EPS);
        assert_close(out[3], -0.5, EPS);
    }

    // -- GELU exact --

    #[test]
    fn test_gelu_exact_zero() {
        let input = [0.0];
        let mut out = [999.0];
        gelu_exact_activate(&input, &mut out);
        assert_close(out[0], 0.0, EPS);
    }

    #[test]
    fn test_gelu_exact_positive() {
        let input = [1.0];
        let mut out = [0.0];
        gelu_exact_activate(&input, &mut out);
        // GELU(1) ≈ 0.8413
        assert_close(out[0], 0.8413, 1e-3);
    }

    #[test]
    fn test_gelu_exact_inplace() {
        let mut data = [0.0, 1.0, -1.0];
        gelu_exact_activate_inplace(&mut data);
        assert_close(data[0], 0.0, EPS);
        assert!(data[1] > 0.5);
        assert!(data[2] < 0.0);
    }

    // -- GELU fast --

    #[test]
    fn test_gelu_fast_close_to_exact() {
        let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
        let mut exact = vec![0.0; input.len()];
        let mut fast = vec![0.0; input.len()];
        gelu_exact_activate(&input, &mut exact);
        gelu_fast_activate(&input, &mut fast);
        for (e, f) in exact.iter().zip(fast.iter()) {
            assert_close(*e, *f, 0.02);
        }
    }

    #[test]
    fn test_gelu_fast_inplace() {
        let mut data = [0.0, 1.0];
        gelu_fast_activate_inplace(&mut data);
        assert_close(data[0], 0.0, EPS);
        assert!(data[1] > 0.5);
    }

    // -- SiLU / Swish --

    #[test]
    fn test_silu_zero() {
        let input = [0.0];
        let mut out = [999.0];
        silu_activate(&input, &mut out);
        assert_close(out[0], 0.0, EPS);
    }

    #[test]
    fn test_silu_known_value() {
        // SiLU(1) = 1/(1+exp(-1)) ≈ 0.7311
        let input = [1.0];
        let mut out = [0.0];
        silu_activate(&input, &mut out);
        assert_close(out[0], 0.7311, 1e-3);
    }

    #[test]
    fn test_silu_inplace() {
        let mut data = [0.0, 1.0, -1.0];
        silu_activate_inplace(&mut data);
        assert_close(data[0], 0.0, EPS);
        assert!(data[1] > 0.0);
        assert!(data[2] < 0.0);
    }

    // -- Sigmoid --

    #[test]
    fn test_sigmoid_bounds() {
        let input = [-100.0, 0.0, 100.0];
        let mut out = [0.0; 3];
        sigmoid_activate(&input, &mut out);
        assert_close(out[0], 0.0, EPS);
        assert_close(out[1], 0.5, EPS);
        assert_close(out[2], 1.0, EPS);
    }

    #[test]
    fn test_sigmoid_inplace() {
        let mut data = [0.0];
        sigmoid_activate_inplace(&mut data);
        assert_close(data[0], 0.5, EPS);
    }

    // -- Tanh --

    #[test]
    fn test_tanh_known_values() {
        let input = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        tanh_activate(&input, &mut out);
        assert_close(out[0], 0.0, EPS);
        assert_close(out[1], 1.0_f32.tanh(), EPS);
        assert_close(out[2], (-1.0_f32).tanh(), EPS);
    }

    #[test]
    fn test_tanh_inplace() {
        let mut data = [0.0, 100.0, -100.0];
        tanh_activate_inplace(&mut data);
        assert_close(data[0], 0.0, EPS);
        assert_close(data[1], 1.0, EPS);
        assert_close(data[2], -1.0, EPS);
    }

    // -- Mish --

    #[test]
    fn test_mish_zero() {
        let input = [0.0];
        let mut out = [999.0];
        mish_activate(&input, &mut out);
        assert_close(out[0], 0.0, EPS);
    }

    #[test]
    fn test_mish_positive() {
        let input = [1.0];
        let mut out = [0.0];
        mish_activate(&input, &mut out);
        // Mish(1) = 1·tanh(ln(1+e)) ≈ 0.8651
        assert_close(out[0], 0.8651, 1e-3);
    }

    #[test]
    fn test_mish_inplace() {
        let mut data = [0.0, 1.0];
        mish_activate_inplace(&mut data);
        assert_close(data[0], 0.0, EPS);
        assert!(data[1] > 0.5);
    }

    // -- HardSwish --

    #[test]
    fn test_hard_swish_regions() {
        let input = [-4.0, -3.0, 0.0, 3.0, 4.0];
        let mut out = [0.0; 5];
        hard_swish_activate(&input, &mut out);
        assert_close(out[0], 0.0, EPS); // below -3
        assert_close(out[1], 0.0, EPS); // boundary
        assert_close(out[2], 0.0, EPS); // midpoint
        assert_close(out[3], 3.0, EPS); // boundary
        assert_close(out[4], 4.0, EPS); // above 3
    }

    #[test]
    fn test_hard_swish_inplace() {
        let mut data = [-4.0, 0.0, 4.0];
        hard_swish_activate_inplace(&mut data);
        assert_close(data[0], 0.0, EPS);
        assert_close(data[1], 0.0, EPS);
        assert_close(data[2], 4.0, EPS);
    }

    // -- HardSigmoid --

    #[test]
    fn test_hard_sigmoid_regions() {
        let input = [-4.0, -3.0, 0.0, 3.0, 4.0];
        let mut out = [0.0; 5];
        hard_sigmoid_activate(&input, &mut out);
        assert_close(out[0], 0.0, EPS);
        assert_close(out[1], 0.0, EPS);
        assert_close(out[2], 0.5, EPS);
        assert_close(out[3], 1.0, EPS);
        assert_close(out[4], 1.0, EPS);
    }

    #[test]
    fn test_hard_sigmoid_inplace() {
        let mut data = [0.0];
        hard_sigmoid_activate_inplace(&mut data);
        assert_close(data[0], 0.5, EPS);
    }

    // -- Edge cases: empty slices --

    #[test]
    fn test_empty_input() {
        let input: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        relu_activate(&input, &mut out);
        silu_activate(&input, &mut out);
        sigmoid_activate(&input, &mut out);
        tanh_activate(&input, &mut out);
        mish_activate(&input, &mut out);
        gelu_exact_activate(&input, &mut out);
        gelu_fast_activate(&input, &mut out);
        hard_swish_activate(&input, &mut out);
        hard_sigmoid_activate(&input, &mut out);
    }

    // -- Edge cases: NaN propagation --

    #[test]
    fn test_nan_propagation() {
        let input = [f32::NAN];
        let mut out = [0.0];

        // ReLU uses f32::max which returns the non-NaN argument per
        // IEEE 754-2008 minNum/maxNum, so NaN → 0.0 is correct.
        relu_activate(&input, &mut out);
        assert_eq!(out[0], 0.0, "ReLU(NaN) = max(0,NaN) = 0");

        silu_activate(&input, &mut out);
        assert!(out[0].is_nan(), "SiLU should propagate NaN");

        sigmoid_activate(&input, &mut out);
        assert!(out[0].is_nan(), "Sigmoid should propagate NaN");

        tanh_activate(&input, &mut out);
        assert!(out[0].is_nan(), "Tanh should propagate NaN");

        gelu_exact_activate(&input, &mut out);
        assert!(out[0].is_nan(), "GELU exact should propagate NaN");

        gelu_fast_activate(&input, &mut out);
        assert!(out[0].is_nan(), "GELU fast should propagate NaN");

        mish_activate(&input, &mut out);
        assert!(out[0].is_nan(), "Mish should propagate NaN");
    }

    // -- Edge cases: Inf handling --

    #[test]
    fn test_inf_handling() {
        let pos_inf = [f32::INFINITY];
        let neg_inf = [f32::NEG_INFINITY];
        let mut out = [0.0];

        relu_activate(&pos_inf, &mut out);
        assert_eq!(out[0], f32::INFINITY);
        relu_activate(&neg_inf, &mut out);
        assert_eq!(out[0], 0.0);

        sigmoid_activate(&pos_inf, &mut out);
        assert_close(out[0], 1.0, EPS);
        sigmoid_activate(&neg_inf, &mut out);
        assert_close(out[0], 0.0, EPS);

        tanh_activate(&pos_inf, &mut out);
        assert_close(out[0], 1.0, EPS);
        tanh_activate(&neg_inf, &mut out);
        assert_close(out[0], -1.0, EPS);

        hard_sigmoid_activate(&pos_inf, &mut out);
        assert_close(out[0], 1.0, EPS);
        hard_sigmoid_activate(&neg_inf, &mut out);
        assert_close(out[0], 0.0, EPS);
    }

    // -- Consistency: in-place vs out-of-place --

    #[test]
    fn test_inplace_matches_out_of_place() {
        let input = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0];
        let n = input.len();

        macro_rules! check_pair {
            ($oop:ident, $ip:ident $(, $arg:expr)*) => {{
                let mut oop_out = vec![0.0; n];
                $oop(&input, &mut oop_out $(, $arg)*);
                let mut ip_data = input.to_vec();
                $ip(&mut ip_data $(, $arg)*);
                for (a, b) in oop_out.iter().zip(ip_data.iter()) {
                    assert_close(*a, *b, EPS);
                }
            }};
        }

        check_pair!(relu_activate, relu_activate_inplace);
        check_pair!(leaky_relu_activate, leaky_relu_activate_inplace, 0.1);
        check_pair!(gelu_exact_activate, gelu_exact_activate_inplace);
        check_pair!(gelu_fast_activate, gelu_fast_activate_inplace);
        check_pair!(silu_activate, silu_activate_inplace);
        check_pair!(sigmoid_activate, sigmoid_activate_inplace);
        check_pair!(tanh_activate, tanh_activate_inplace);
        check_pair!(mish_activate, mish_activate_inplace);
        check_pair!(hard_swish_activate, hard_swish_activate_inplace);
        check_pair!(hard_sigmoid_activate, hard_sigmoid_activate_inplace);
    }
}
