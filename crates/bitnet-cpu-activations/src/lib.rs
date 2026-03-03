//! CPU activation function kernels.
//!
//! Provides 15 activation functions commonly used in neural networks,
//! with elementwise apply, in-place, and derivative variants.

use std::f32::consts::PI;

// ── Activation type enum ────────────────────────────────────────────

/// Supported activation function types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: x if x > 0, else alpha * x
    LeakyReLU(f32),
    /// Gaussian Error Linear Unit (erf approximation)
    GELU,
    /// GELU with tanh approximation
    GELUTanh,
    /// Sigmoid Linear Unit: x * sigmoid(x)
    SiLU,
    /// Swish with beta parameter: x * sigmoid(beta * x)
    Swish(f32),
    /// Logistic sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Hard sigmoid: clamp(x/6 + 0.5, 0, 1)
    HardSigmoid,
    /// Hard swish: x * hard_sigmoid(x)
    HardSwish,
    /// Mish: x * tanh(softplus(x))
    Mish,
    /// Softplus: ln(1 + exp(x))
    Softplus,
    /// Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1)
    ELU(f32),
    /// Scaled ELU: lambda * (x if x > 0, else alpha * (exp(x) - 1))
    SELU,
    /// Quick GELU: x * sigmoid(1.702 * x)
    QuickGELU,
}

// SELU constants (Klambauer et al., 2017)
const SELU_ALPHA: f32 = 1.6732632;
const SELU_LAMBDA: f32 = 1.050_701;

// ── Individual activation functions ─────────────────────────────────

/// ReLU: max(0, x), preserving NaN
#[inline]
pub fn relu(x: f32) -> f32 {
    if x.is_nan() { x } else { x.max(0.0) }
}

/// Leaky ReLU: x if x >= 0, else alpha * x
#[inline]
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x.is_nan() || x >= 0.0 { x } else { alpha * x }
}

/// Sigmoid: 1 / (1 + exp(-x))
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation (delegates to std)
#[inline]
pub fn tanh_act(x: f32) -> f32 {
    x.tanh()
}

/// GELU using the erf-based formula: x * 0.5 * (1 + erf(x / sqrt(2)))
#[inline]
pub fn gelu(x: f32) -> f32 {
    // Use the tanh approximation of erf for f32:
    // erf(a) ≈ tanh(sqrt(2/pi) * (a + 0.044715 * a^3))
    // So GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // But the "true" GELU uses erf directly. We use libm's erff via f64.
    let xd = x as f64;
    let cdf = 0.5 * (1.0 + libm::erf(xd / std::f64::consts::SQRT_2));
    (xd * cdf) as f32
}

/// GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// SiLU (Swish-1): x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Swish: x * sigmoid(beta * x)
#[inline]
pub fn swish(x: f32, beta: f32) -> f32 {
    x * sigmoid(beta * x)
}

/// Hard sigmoid: clamp(x/6 + 0.5, 0, 1)
#[inline]
pub fn hard_sigmoid(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    (x / 6.0 + 0.5).clamp(0.0, 1.0)
}

/// Hard swish: x * hard_sigmoid(x)
#[inline]
pub fn hard_swish(x: f32) -> f32 {
    x * hard_sigmoid(x)
}

/// Softplus: ln(1 + exp(x)), with numerical stability
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // For large x, softplus ≈ x
    } else if x < -20.0 {
        0.0 // For very negative x, softplus ≈ 0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Mish: x * tanh(softplus(x))
#[inline]
pub fn mish(x: f32) -> f32 {
    x * softplus(x).tanh()
}

/// ELU: x if x > 0, else alpha * (exp(x) - 1)
#[inline]
pub fn elu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

/// SELU: lambda * ELU(x, alpha) with fixed constants
#[inline]
pub fn selu(x: f32) -> f32 {
    SELU_LAMBDA * elu(x, SELU_ALPHA)
}

/// Quick GELU: x * sigmoid(1.702 * x)
#[inline]
pub fn quick_gelu(x: f32) -> f32 {
    x * sigmoid(1.702 * x)
}

/// Softplus with configurable beta: (1/beta) * ln(1 + exp(beta * x))
#[inline]
pub fn softplus_beta(x: f32, beta: f32) -> f32 {
    let bx = beta * x;
    if bx > 20.0 {
        x
    } else if bx < -20.0 {
        0.0
    } else {
        (1.0 + bx.exp()).ln() / beta
    }
}

// ── Slice-based convenience functions ───────────────────────────────

/// GELU (erf-based) applied elementwise to a slice.
pub fn gelu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| gelu(v)).collect()
}

/// Fast GELU approximation (tanh-based) applied elementwise.
pub fn gelu_approx_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| gelu_tanh(v)).collect()
}

/// SiLU/Swish applied elementwise to a slice.
pub fn silu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| silu(v)).collect()
}

/// Mish applied elementwise to a slice.
pub fn mish_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| mish(v)).collect()
}

/// Softplus with beta applied elementwise to a slice.
pub fn softplus_vec(x: &[f32], beta: f32) -> Vec<f32> {
    x.iter().map(|&v| softplus_beta(v, beta)).collect()
}

/// Hard sigmoid applied elementwise to a slice.
pub fn hard_sigmoid_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| hard_sigmoid(v)).collect()
}

/// Hard swish applied elementwise to a slice.
pub fn hard_swish_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| hard_swish(v)).collect()
}

/// Leaky ReLU applied elementwise to a slice.
pub fn leaky_relu_vec(x: &[f32], negative_slope: f32) -> Vec<f32> {
    x.iter().map(|&v| leaky_relu(v, negative_slope)).collect()
}

/// ELU applied elementwise to a slice.
pub fn elu_vec(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter().map(|&v| elu(v, alpha)).collect()
}

// ── Named in-place variants ─────────────────────────────────────────

/// GELU activation applied in-place.
pub fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu(*v);
    }
}

/// SiLU activation applied in-place.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// ReLU activation applied in-place.
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = relu(*v);
    }
}

// ── Dispatcher ──────────────────────────────────────────────────────

/// Apply an activation function elementwise, returning a new vector.
///
/// This is the primary dispatch entry point for activation functions.
pub fn apply_activation(x: &[f32], activation: ActivationType) -> Vec<f32> {
    activate(x, activation)
}

// ── Dispatch helpers ────────────────────────────────────────────────

/// Apply a single activation function to a scalar value.
#[inline]
fn apply_one(x: f32, activation: ActivationType) -> f32 {
    match activation {
        ActivationType::ReLU => relu(x),
        ActivationType::LeakyReLU(alpha) => leaky_relu(x, alpha),
        ActivationType::GELU => gelu(x),
        ActivationType::GELUTanh => gelu_tanh(x),
        ActivationType::SiLU => silu(x),
        ActivationType::Swish(beta) => swish(x, beta),
        ActivationType::Sigmoid => sigmoid(x),
        ActivationType::Tanh => tanh_act(x),
        ActivationType::HardSigmoid => hard_sigmoid(x),
        ActivationType::HardSwish => hard_swish(x),
        ActivationType::Mish => mish(x),
        ActivationType::Softplus => softplus(x),
        ActivationType::ELU(alpha) => elu(x, alpha),
        ActivationType::SELU => selu(x),
        ActivationType::QuickGELU => quick_gelu(x),
    }
}

// ── Public vectorised API ───────────────────────────────────────────

/// Apply `activation` elementwise, returning a new vector.
pub fn activate(input: &[f32], activation: ActivationType) -> Vec<f32> {
    input.iter().map(|&x| apply_one(x, activation)).collect()
}

/// Apply `activation` elementwise in-place.
pub fn activate_inplace(input: &mut [f32], activation: ActivationType) {
    for x in input.iter_mut() {
        *x = apply_one(*x, activation);
    }
}

/// Compute the derivative of `activation` at each element.
///
/// For most activations the derivative is with respect to the
/// *pre-activation* input (i.e. the value before the activation was
/// applied).
pub fn activate_derivative(input: &[f32], activation: ActivationType) -> Vec<f32> {
    input.iter().map(|&x| derivative_one(x, activation)).collect()
}

/// Derivative of a single activation at a scalar value.
#[inline]
fn derivative_one(x: f32, activation: ActivationType) -> f32 {
    match activation {
        ActivationType::ReLU => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        ActivationType::LeakyReLU(alpha) => {
            if x >= 0.0 {
                1.0
            } else {
                alpha
            }
        }
        ActivationType::GELU => {
            // d/dx GELU ≈ Φ(x) + x·φ(x)
            let xd = x as f64;
            let sqrt2 = std::f64::consts::SQRT_2;
            let phi = 0.5 * (1.0 + libm::erf(xd / sqrt2)); // CDF
            let pdf = (-(xd * xd) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
            (phi + xd * pdf) as f32
        }
        ActivationType::GELUTanh => {
            // Numerical derivative via central differences
            let h = 1e-4_f32;
            (gelu_tanh(x + h) - gelu_tanh(x - h)) / (2.0 * h)
        }
        ActivationType::SiLU => {
            let s = sigmoid(x);
            s + x * s * (1.0 - s)
        }
        ActivationType::Swish(beta) => {
            let s = sigmoid(beta * x);
            s + beta * x * s * (1.0 - s)
        }
        ActivationType::Sigmoid => {
            let s = sigmoid(x);
            s * (1.0 - s)
        }
        ActivationType::Tanh => {
            let t = x.tanh();
            1.0 - t * t
        }
        ActivationType::HardSigmoid => {
            if !(-3.0..=3.0).contains(&x) {
                0.0
            } else {
                1.0 / 6.0
            }
        }
        ActivationType::HardSwish => {
            if x <= -3.0 {
                0.0
            } else if x >= 3.0 {
                1.0
            } else {
                x / 3.0 + 0.5
            }
        }
        ActivationType::Mish => {
            let h = 1e-4_f32;
            (mish(x + h) - mish(x - h)) / (2.0 * h)
        }
        ActivationType::Softplus => sigmoid(x),
        ActivationType::ELU(alpha) => {
            if x > 0.0 {
                1.0
            } else {
                alpha * x.exp()
            }
        }
        ActivationType::SELU => {
            if x > 0.0 {
                SELU_LAMBDA
            } else {
                SELU_LAMBDA * SELU_ALPHA * x.exp()
            }
        }
        ActivationType::QuickGELU => {
            let s = sigmoid(1.702 * x);
            s + 1.702 * x * s * (1.0 - s)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < tol
    }

    fn numerical_derivative(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    // ── ReLU ──

    #[test]
    fn test_relu_positive() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(0.1), 0.1);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(-0.1), 0.0);
    }

    #[test]
    fn test_relu_zero() {
        assert_eq!(relu(0.0), 0.0);
    }

    // ── LeakyReLU ──

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(2.0, 0.01), 2.0);
        assert!(approx_eq(leaky_relu(-2.0, 0.01), -0.02, 1e-6));
        assert_eq!(leaky_relu(0.0, 0.01), 0.0);
    }

    // ── Sigmoid ──

    #[test]
    fn test_sigmoid_at_zero() {
        assert!(approx_eq(sigmoid(0.0), 0.5, 1e-6));
    }

    #[test]
    fn test_sigmoid_extremes() {
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let x = 2.5;
        assert!(approx_eq(sigmoid(x) + sigmoid(-x), 1.0, 1e-6));
    }

    // ── Tanh ──

    #[test]
    fn test_tanh_at_zero() {
        assert!(approx_eq(tanh_act(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_tanh_extremes() {
        assert!(tanh_act(100.0) > 0.999);
        assert!(tanh_act(-100.0) < -0.999);
    }

    // ── GELU ──

    #[test]
    fn test_gelu_at_zero() {
        assert!(approx_eq(gelu(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_gelu_positive_region() {
        // GELU(x) ≈ x for large positive x
        assert!(gelu(5.0) > 4.99);
    }

    #[test]
    fn test_gelu_negative_region() {
        // GELU is slightly negative for small negative inputs
        assert!(gelu(-0.5) < 0.0);
    }

    // ── GELUTanh ──

    #[test]
    fn test_gelu_tanh_close_to_gelu() {
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert!(
                approx_eq(gelu_tanh(x), gelu(x), 0.02),
                "GELUTanh({x}) = {} vs GELU({x}) = {}",
                gelu_tanh(x),
                gelu(x)
            );
        }
    }

    // ── SiLU ──

    #[test]
    fn test_silu_at_zero() {
        assert!(approx_eq(silu(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_silu_positive() {
        // SiLU(x) ≈ x for large positive x
        assert!(silu(10.0) > 9.99);
    }

    #[test]
    fn test_silu_negative() {
        // SiLU has a small negative trough around x ≈ -1.28
        assert!(silu(-1.28) < 0.0);
    }

    // ── Swish ──

    #[test]
    fn test_swish_beta_one_equals_silu() {
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert!(approx_eq(swish(x, 1.0), silu(x), 1e-6));
        }
    }

    // ── HardSigmoid ──

    #[test]
    fn test_hard_sigmoid_clamp() {
        assert!(approx_eq(hard_sigmoid(0.0), 0.5, 1e-6));
        assert_eq!(hard_sigmoid(-10.0), 0.0);
        assert_eq!(hard_sigmoid(10.0), 1.0);
    }

    // ── HardSwish ──

    #[test]
    fn test_hard_swish_at_zero() {
        assert!(approx_eq(hard_swish(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_hard_swish_extremes() {
        assert_eq!(hard_swish(-4.0), 0.0);
        assert!(approx_eq(hard_swish(4.0), 4.0, 1e-6));
    }

    // ── Softplus ──

    #[test]
    fn test_softplus_positive() {
        // softplus(0) = ln(2)
        assert!(approx_eq(softplus(0.0), 2.0_f32.ln(), 1e-5));
    }

    #[test]
    fn test_softplus_large_input() {
        // For large x, softplus ≈ x
        assert!(approx_eq(softplus(50.0), 50.0, 1e-3));
    }

    // ── Mish ──

    #[test]
    fn test_mish_at_zero() {
        assert!(approx_eq(mish(0.0), 0.0, 1e-5));
    }

    #[test]
    fn test_mish_positive() {
        // Mish(x) ≈ x for large positive x
        assert!(mish(10.0) > 9.99);
    }

    // ── ELU ──

    #[test]
    fn test_elu_positive() {
        assert_eq!(elu(2.0, 1.0), 2.0);
    }

    #[test]
    fn test_elu_negative() {
        let val = elu(-1.0, 1.0);
        assert!(val < 0.0 && val > -1.0);
    }

    #[test]
    fn test_elu_zero() {
        assert!(approx_eq(elu(0.0, 1.0), 0.0, 1e-6));
    }

    // ── SELU ──

    #[test]
    fn test_selu_positive() {
        assert!(approx_eq(selu(1.0), SELU_LAMBDA, 1e-5));
    }

    #[test]
    fn test_selu_zero() {
        // SELU(0) = lambda * alpha * (exp(0)-1) = 0
        assert!(approx_eq(selu(0.0), 0.0, 1e-5));
    }

    // ── QuickGELU ──

    #[test]
    fn test_quick_gelu_at_zero() {
        assert!(approx_eq(quick_gelu(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_quick_gelu_close_to_gelu() {
        for x in [-1.0, 0.0, 1.0, 2.0] {
            assert!(
                approx_eq(quick_gelu(x), gelu(x), 0.05),
                "QuickGELU({x}) = {} vs GELU({x}) = {}",
                quick_gelu(x),
                gelu(x)
            );
        }
    }

    // ── activate / activate_inplace ──

    #[test]
    fn test_activate_relu_vec() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let out = activate(&input, ActivationType::ReLU);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_activate_inplace_sigmoid() {
        let mut data = vec![0.0, 1.0, -1.0];
        let expected = activate(&data, ActivationType::Sigmoid);
        activate_inplace(&mut data, ActivationType::Sigmoid);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_activate_empty() {
        let empty: Vec<f32> = vec![];
        assert!(activate(&empty, ActivationType::ReLU).is_empty());
    }

    #[test]
    fn test_activate_inplace_empty() {
        let mut empty: Vec<f32> = vec![];
        activate_inplace(&mut empty, ActivationType::ReLU);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_activate_matches_individual_functions() {
        let input = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
        let types_and_fns: Vec<(ActivationType, Box<dyn Fn(f32) -> f32>)> = vec![
            (ActivationType::ReLU, Box::new(relu)),
            (ActivationType::LeakyReLU(0.01), Box::new(|x| leaky_relu(x, 0.01))),
            (ActivationType::GELU, Box::new(gelu)),
            (ActivationType::GELUTanh, Box::new(gelu_tanh)),
            (ActivationType::SiLU, Box::new(silu)),
            (ActivationType::Swish(1.0), Box::new(|x| swish(x, 1.0))),
            (ActivationType::Sigmoid, Box::new(sigmoid)),
            (ActivationType::Tanh, Box::new(tanh_act)),
            (ActivationType::HardSigmoid, Box::new(hard_sigmoid)),
            (ActivationType::HardSwish, Box::new(hard_swish)),
            (ActivationType::Mish, Box::new(mish)),
            (ActivationType::Softplus, Box::new(softplus)),
            (ActivationType::ELU(1.0), Box::new(|x| elu(x, 1.0))),
            (ActivationType::SELU, Box::new(selu)),
            (ActivationType::QuickGELU, Box::new(quick_gelu)),
        ];
        for (act_type, f) in &types_and_fns {
            let via_dispatch = activate(&input, *act_type);
            let via_fn: Vec<f32> = input.iter().map(|&x| f(x)).collect();
            for (i, (a, b)) in via_dispatch.iter().zip(via_fn.iter()).enumerate() {
                assert!(approx_eq(*a, *b, 1e-5), "{act_type:?} mismatch at {i}: {a} vs {b}");
            }
        }
    }

    // ── NaN propagation ──

    #[test]
    fn test_nan_propagation_all_activations() {
        let activations = [
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
        for act in &activations {
            let out = activate(&[f32::NAN], *act);
            assert!(out[0].is_nan(), "{act:?} did not propagate NaN");
        }
    }

    // ── Infinity handling ──

    #[test]
    fn test_inf_handling() {
        // ReLU(+inf) = +inf, ReLU(-inf) = 0
        assert_eq!(relu(f32::INFINITY), f32::INFINITY);
        assert_eq!(relu(f32::NEG_INFINITY), 0.0);

        // Sigmoid(+inf) → 1, Sigmoid(-inf) → 0
        assert!(approx_eq(sigmoid(f32::INFINITY), 1.0, 1e-5));
        assert!(approx_eq(sigmoid(f32::NEG_INFINITY), 0.0, 1e-5));

        // Tanh(+inf) → 1, Tanh(-inf) → -1
        assert!(approx_eq(tanh_act(f32::INFINITY), 1.0, 1e-5));
        assert!(approx_eq(tanh_act(f32::NEG_INFINITY), -1.0, 1e-5));
    }

    // ── Zeros ──

    #[test]
    fn test_all_activations_at_zero() {
        // Most activations at zero should be deterministic
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(leaky_relu(0.0, 0.01), 0.0);
        assert!(approx_eq(gelu(0.0), 0.0, 1e-6));
        assert!(approx_eq(gelu_tanh(0.0), 0.0, 1e-6));
        assert!(approx_eq(silu(0.0), 0.0, 1e-6));
        assert!(approx_eq(sigmoid(0.0), 0.5, 1e-6));
        assert!(approx_eq(tanh_act(0.0), 0.0, 1e-6));
        assert!(approx_eq(hard_sigmoid(0.0), 0.5, 1e-6));
        assert!(approx_eq(hard_swish(0.0), 0.0, 1e-6));
        assert!(approx_eq(mish(0.0), 0.0, 1e-5));
        assert!(approx_eq(softplus(0.0), 2.0_f32.ln(), 1e-5));
        assert!(approx_eq(elu(0.0, 1.0), 0.0, 1e-6));
        assert!(approx_eq(selu(0.0), 0.0, 1e-5));
        assert!(approx_eq(quick_gelu(0.0), 0.0, 1e-6));
    }

    // ── Derivative tests ──

    #[test]
    fn test_relu_derivative() {
        let d = activate_derivative(&[-1.0, 0.0, 1.0], ActivationType::ReLU);
        assert_eq!(d, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_derivative() {
        // sigmoid'(0) = sigmoid(0)*(1-sigmoid(0)) = 0.25
        let d = activate_derivative(&[0.0], ActivationType::Sigmoid);
        assert!(approx_eq(d[0], 0.25, 1e-5));
    }

    #[test]
    fn test_tanh_derivative() {
        // tanh'(0) = 1 - tanh(0)^2 = 1
        let d = activate_derivative(&[0.0], ActivationType::Tanh);
        assert!(approx_eq(d[0], 1.0, 1e-5));
    }

    #[test]
    fn test_softplus_derivative_is_sigmoid() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let d = activate_derivative(&input, ActivationType::Softplus);
        let s = activate(&input, ActivationType::Sigmoid);
        for (a, b) in d.iter().zip(s.iter()) {
            assert!(approx_eq(*a, *b, 1e-5));
        }
    }

    #[test]
    fn test_derivative_numerical_check() {
        // Verify analytical derivatives against numerical for several types
        let test_points = [-2.0_f32, -0.5, 0.5, 2.0];
        let activations = [
            ActivationType::ReLU,
            ActivationType::SiLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::ELU(1.0),
            ActivationType::SELU,
            ActivationType::QuickGELU,
        ];
        let h = 1e-4;
        for act in &activations {
            for &x in &test_points {
                // Skip ReLU at discontinuity near zero
                if matches!(act, ActivationType::ReLU) && x.abs() < 0.1 {
                    continue;
                }
                let analytical = activate_derivative(&[x], *act)[0];
                let numerical = numerical_derivative(|v| activate(&[v], *act)[0], x, h);
                assert!(
                    approx_eq(analytical, numerical, 1e-2),
                    "{act:?} derivative at {x}: analytical={analytical}, \
                     numerical={numerical}"
                );
            }
        }
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let alpha = 0.1;
        let d = activate_derivative(&[-1.0, 0.0, 1.0], ActivationType::LeakyReLU(alpha));
        assert!(approx_eq(d[0], alpha, 1e-5));
        assert!(approx_eq(d[1], 1.0, 1e-5));
        assert!(approx_eq(d[2], 1.0, 1e-5));
    }

    #[test]
    fn test_hard_sigmoid_derivative() {
        let d = activate_derivative(&[-5.0, 0.0, 5.0], ActivationType::HardSigmoid);
        assert!(approx_eq(d[0], 0.0, 1e-5)); // outside [-3,3]
        assert!(approx_eq(d[1], 1.0 / 6.0, 1e-5)); // inside
        assert!(approx_eq(d[2], 0.0, 1e-5)); // outside
    }

    #[test]
    fn test_elu_derivative() {
        let d = activate_derivative(&[-1.0, 0.5], ActivationType::ELU(1.0));
        assert!(approx_eq(d[0], (-1.0_f32).exp(), 1e-5));
        assert!(approx_eq(d[1], 1.0, 1e-5));
    }

    // ── Edge cases ──

    #[test]
    fn test_large_input_stability() {
        let large = vec![100.0, -100.0, 1000.0, -1000.0];
        // These should not panic or produce NaN (except where mathematically
        // expected, e.g. ELU(-1000) ≈ -alpha which is fine)
        for act in [
            ActivationType::ReLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::HardSigmoid,
            ActivationType::HardSwish,
            ActivationType::Softplus,
            ActivationType::SELU,
        ] {
            let out = activate(&large, act);
            for (i, v) in out.iter().enumerate() {
                assert!(!v.is_nan(), "{act:?} produced NaN at index {i} for input {}", large[i]);
            }
        }
    }

    #[test]
    fn test_negative_inputs() {
        let neg = vec![-0.001, -0.01, -0.1, -1.0, -10.0];
        let relu_out = activate(&neg, ActivationType::ReLU);
        assert!(relu_out.iter().all(|&v| v == 0.0));

        let leaky_out = activate(&neg, ActivationType::LeakyReLU(0.1));
        assert!(leaky_out.iter().all(|&v| v < 0.0));
    }

    #[test]
    fn test_monotonicity_sigmoid() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = activate(&input, ActivationType::Sigmoid);
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - 1e-6, "Sigmoid not monotonic at i={i}");
        }
    }

    #[test]
    fn test_monotonicity_relu() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = activate(&input, ActivationType::ReLU);
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - 1e-6, "ReLU not monotonic at i={i}");
        }
    }

    // ── softplus_beta tests ──

    #[test]
    fn test_softplus_beta_one_matches_standard() {
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0, 10.0] {
            assert!(
                approx_eq(softplus_beta(x, 1.0), softplus(x), 1e-5),
                "softplus_beta(1.0) mismatch at {x}"
            );
        }
    }

    #[test]
    fn test_softplus_beta_scaling() {
        // Higher beta → steeper transition (closer to ReLU)
        let x = 0.5;
        let low = softplus_beta(x, 1.0);
        let high = softplus_beta(x, 10.0);
        assert!(high < low, "Higher beta should yield sharper (lower) output near origin");
    }

    #[test]
    fn test_softplus_beta_numerical_stability() {
        // Large and small inputs should not panic or produce NaN
        assert!(!softplus_beta(100.0, 1.0).is_nan());
        assert!(!softplus_beta(-100.0, 1.0).is_nan());
        assert!(!softplus_beta(100.0, 5.0).is_nan());
        assert!(!softplus_beta(-100.0, 5.0).is_nan());
        assert!(approx_eq(softplus_beta(100.0, 1.0), 100.0, 1e-3));
        assert!(approx_eq(softplus_beta(-100.0, 1.0), 0.0, 1e-3));
    }

    // ── Slice-based function tests ──

    #[test]
    fn test_gelu_vec_matches_scalar() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let vec_out = gelu_vec(&input);
        let scalar_out: Vec<f32> = input.iter().map(|&x| gelu(x)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_gelu_approx_vec_matches_gelu_tanh() {
        let input = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
        let vec_out = gelu_approx_vec(&input);
        let scalar_out: Vec<f32> = input.iter().map(|&x| gelu_tanh(x)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_silu_vec_matches_scalar() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let vec_out = silu_vec(&input);
        let scalar_out: Vec<f32> = input.iter().map(|&x| silu(x)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_mish_vec_matches_scalar() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let vec_out = mish_vec(&input);
        let scalar_out: Vec<f32> = input.iter().map(|&x| mish(x)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_softplus_vec_matches_scalar() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let vec_out = softplus_vec(&input, 1.0);
        let scalar_out: Vec<f32> = input.iter().map(|&x| softplus_beta(x, 1.0)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_hard_sigmoid_vec_matches_scalar() {
        let input = vec![-5.0, -3.0, 0.0, 3.0, 5.0];
        let vec_out = hard_sigmoid_vec(&input);
        let scalar_out: Vec<f32> = input.iter().map(|&x| hard_sigmoid(x)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_hard_swish_vec_matches_scalar() {
        let input = vec![-5.0, -3.0, 0.0, 3.0, 5.0];
        let vec_out = hard_swish_vec(&input);
        let scalar_out: Vec<f32> = input.iter().map(|&x| hard_swish(x)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_leaky_relu_vec_matches_scalar() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let vec_out = leaky_relu_vec(&input, 0.01);
        let scalar_out: Vec<f32> = input.iter().map(|&x| leaky_relu(x, 0.01)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_elu_vec_matches_scalar() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let vec_out = elu_vec(&input, 1.0);
        let scalar_out: Vec<f32> = input.iter().map(|&x| elu(x, 1.0)).collect();
        for (a, b) in vec_out.iter().zip(scalar_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    // ── In-place variant tests ──

    #[test]
    fn test_gelu_inplace_matches_allocating() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = gelu_vec(&input);
        let mut buf = input;
        gelu_inplace(&mut buf);
        for (a, b) in buf.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_silu_inplace_matches_allocating() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = silu_vec(&input);
        let mut buf = input;
        silu_inplace(&mut buf);
        for (a, b) in buf.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_relu_inplace_matches_allocating() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = activate(&input, ActivationType::ReLU);
        let mut buf = input;
        relu_inplace(&mut buf);
        assert_eq!(buf, expected);
    }

    // ── apply_activation dispatcher test ──

    #[test]
    fn test_apply_activation_matches_activate() {
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        for act in [
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::SiLU,
            ActivationType::Mish,
            ActivationType::Softplus,
            ActivationType::ELU(1.0),
        ] {
            let a = activate(&input, act);
            let b = apply_activation(&input, act);
            assert_eq!(a, b, "apply_activation mismatch for {act:?}");
        }
    }

    // ── Monotonicity tests for additional activations ──

    #[test]
    fn test_monotonicity_softplus() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = activate(&input, ActivationType::Softplus);
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - 1e-6, "Softplus not monotonic at i={i}");
        }
    }

    #[test]
    fn test_monotonicity_elu() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = activate(&input, ActivationType::ELU(1.0));
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - 1e-6, "ELU not monotonic at i={i}");
        }
    }

    #[test]
    fn test_monotonicity_leaky_relu() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = activate(&input, ActivationType::LeakyReLU(0.01));
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - 1e-6, "LeakyReLU not monotonic at i={i}");
        }
    }

    #[test]
    fn test_monotonicity_hard_sigmoid() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = activate(&input, ActivationType::HardSigmoid);
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - 1e-6, "HardSigmoid not monotonic at i={i}");
        }
    }

    // ── Boundary and shape tests ──

    #[test]
    fn test_mish_bounded_below() {
        // Mish has a global minimum ≈ -0.31 near x ≈ -1.19
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.05).collect();
        let out = mish_vec(&input);
        for &v in &out {
            assert!(v >= -0.32, "Mish should be bounded below by ≈ -0.31, got {v}");
        }
    }

    #[test]
    fn test_hard_swish_matches_relu_for_large() {
        // For x >> 3, hard_swish(x) ≈ x
        for x in [10.0, 50.0, 100.0] {
            assert!(approx_eq(hard_swish(x), x, 1e-5), "hard_swish({x}) should ≈ x");
        }
    }

    #[test]
    fn test_elu_alpha_zero_is_relu() {
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let elu_out = elu_vec(&input, 0.0);
        let relu_out = activate(&input, ActivationType::ReLU);
        for (a, b) in elu_out.iter().zip(relu_out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6), "ELU(alpha=0) should match ReLU");
        }
    }

    #[test]
    fn test_gelu_approximation_quality() {
        // gelu_approx should be within 0.02 of true GELU for typical range
        let input: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
        let exact = gelu_vec(&input);
        let approx = gelu_approx_vec(&input);
        for (i, (&e, &a)) in exact.iter().zip(approx.iter()).enumerate() {
            assert!(
                approx_eq(e, a, 0.02),
                "GELU approx too far from exact at index {i}: exact={e}, approx={a}"
            );
        }
    }

    #[test]
    fn test_vec_functions_empty_input() {
        let empty: &[f32] = &[];
        assert!(gelu_vec(empty).is_empty());
        assert!(gelu_approx_vec(empty).is_empty());
        assert!(silu_vec(empty).is_empty());
        assert!(mish_vec(empty).is_empty());
        assert!(softplus_vec(empty, 1.0).is_empty());
        assert!(hard_sigmoid_vec(empty).is_empty());
        assert!(hard_swish_vec(empty).is_empty());
        assert!(leaky_relu_vec(empty, 0.01).is_empty());
        assert!(elu_vec(empty, 1.0).is_empty());
    }

    #[test]
    fn test_inplace_empty_input() {
        let mut empty: Vec<f32> = vec![];
        gelu_inplace(&mut empty);
        silu_inplace(&mut empty);
        relu_inplace(&mut empty);
        assert!(empty.is_empty());
    }
}
