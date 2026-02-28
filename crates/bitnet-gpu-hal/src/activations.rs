//! Activation functions for GPU inference.
//!
//! CPU reference implementations and GPU kernel source strings
//! for common neural network activation functions.

use core::f32::consts::{FRAC_2_SQRT_PI, SQRT_2};

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    ReLU,
    GeLU,
    /// Tanh approximation of `GeLU`.
    GeLUApprox,
    /// Swish: x * sigmoid(x).
    SiLU,
    Sigmoid,
    Tanh,
    Softplus,
    /// x * tanh(softplus(x)).
    Mish,
}

/// All activation variants for iteration.
pub const ALL_ACTIVATIONS: &[Activation] = &[
    Activation::ReLU,
    Activation::GeLU,
    Activation::GeLUApprox,
    Activation::SiLU,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Softplus,
    Activation::Mish,
];

impl Activation {
    /// Apply activation to a single value (CPU reference).
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::GeLU => {
                // Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
                x * 0.5 * (1.0 + erf(x / SQRT_2))
            }
            Self::GeLUApprox => {
                // Tanh approximation:
                // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let c = (FRAC_2_SQRT_PI * SQRT_2 * 0.5) * (0.044_715 * x * x).mul_add(x, x);
                0.5 * x * (1.0 + c.tanh())
            }
            Self::SiLU => x * sigmoid(x),
            Self::Sigmoid => sigmoid(x),
            Self::Tanh => x.tanh(),
            Self::Softplus => softplus(x),
            Self::Mish => x * softplus(x).tanh(),
        }
    }

    /// Apply activation to a slice in-place.
    pub fn apply_inplace(&self, data: &mut [f32]) {
        for v in data.iter_mut() {
            *v = self.apply(*v);
        }
    }

    /// Apply activation, writing to output slice.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != output.len()`.
    pub fn apply_batch(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "input and output slices must have equal length");
        for (o, &x) in output.iter_mut().zip(input.iter()) {
            *o = self.apply(x);
        }
    }

    /// `OpenCL` kernel source for this activation.
    pub const fn opencl_source(&self) -> &'static str {
        match self {
            Self::ReLU => include_str!("kernels/opencl/relu.cl"),
            Self::GeLU => include_str!("kernels/opencl/gelu.cl"),
            Self::GeLUApprox => {
                include_str!("kernels/opencl/gelu_approx.cl")
            }
            Self::SiLU => include_str!("kernels/opencl/silu.cl"),
            Self::Sigmoid => {
                include_str!("kernels/opencl/sigmoid.cl")
            }
            Self::Tanh => include_str!("kernels/opencl/tanh.cl"),
            Self::Softplus => {
                include_str!("kernels/opencl/softplus.cl")
            }
            Self::Mish => include_str!("kernels/opencl/mish.cl"),
        }
    }

    /// WGSL kernel source for this activation.
    pub const fn wgsl_source(&self) -> &'static str {
        match self {
            Self::ReLU => include_str!("kernels/wgsl/relu.wgsl"),
            Self::GeLU => include_str!("kernels/wgsl/gelu.wgsl"),
            Self::GeLUApprox => {
                include_str!("kernels/wgsl/gelu_approx.wgsl")
            }
            Self::SiLU => include_str!("kernels/wgsl/silu.wgsl"),
            Self::Sigmoid => {
                include_str!("kernels/wgsl/sigmoid.wgsl")
            }
            Self::Tanh => include_str!("kernels/wgsl/tanh.wgsl"),
            Self::Softplus => {
                include_str!("kernels/wgsl/softplus.wgsl")
            }
            Self::Mish => include_str!("kernels/wgsl/mish.wgsl"),
        }
    }

    /// Derivative of the activation (for backward pass reference).
    #[inline]
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::GeLU => {
                let cdf = 0.5 * (1.0 + erf(x / SQRT_2));
                let pdf = (-0.5 * x * x).exp() / (2.0 * core::f32::consts::PI).sqrt();
                x.mul_add(pdf, cdf)
            }
            Self::GeLUApprox => {
                let a = 0.044_715_f32;
                let s = (FRAC_2_SQRT_PI * SQRT_2 * 0.5) * (a * x * x).mul_add(x, x);
                let t = s.tanh();
                let dt = t.mul_add(-t, 1.0);
                let ds = (FRAC_2_SQRT_PI * SQRT_2 * 0.5) * (3.0 * a * x).mul_add(x, 1.0);
                0.5_f32.mul_add(1.0 + t, 0.5 * x * dt * ds)
            }
            Self::SiLU => {
                let s = sigmoid(x);
                (x * s).mul_add(1.0 - s, s)
            }
            Self::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            Self::Tanh => {
                let t = x.tanh();
                t.mul_add(-t, 1.0)
            }
            Self::Softplus => sigmoid(x),
            Self::Mish => {
                let sp = softplus(x);
                let t = sp.tanh();
                let s = sigmoid(x);
                // d/dx [x * tanh(softplus(x))]
                // = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
                (x * t.mul_add(-t, 1.0)).mul_add(s, t)
            }
        }
    }

    /// Human-readable name.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ReLU => "ReLU",
            Self::GeLU => "GeLU",
            Self::GeLUApprox => "GeLU (tanh approx)",
            Self::SiLU => "SiLU",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::Softplus => "Softplus",
            Self::Mish => "Mish",
        }
    }
}

/// Fused activation + residual add: `output[i] = act(input[i] + residual[i])`.
///
/// # Panics
///
/// Panics if slices have different lengths.
pub fn fused_add_activation(input: &[f32], residual: &[f32], output: &mut [f32], act: Activation) {
    assert_eq!(input.len(), residual.len());
    assert_eq!(input.len(), output.len());
    for i in 0..input.len() {
        output[i] = act.apply(input[i] + residual[i]);
    }
}

// ── helpers ──────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    // Numerically stable: for large x, softplus(x) ≈ x
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        x.exp().ln_1p()
    }
}

/// Approximate error function (Abramowitz & Stegun 7.1.26).
#[inline]
#[allow(clippy::excessive_precision)]
fn erf(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / 0.327_591_1_f32.mul_add(x, 1.0);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol || (a.is_nan() && b.is_nan())
    }

    // ── ReLU tests ───────────────────────────────────────────

    #[test]
    fn relu_positive() {
        assert_eq!(Activation::ReLU.apply(3.0), 3.0);
    }

    #[test]
    fn relu_negative() {
        assert_eq!(Activation::ReLU.apply(-2.0), 0.0);
    }

    #[test]
    fn relu_zero() {
        assert_eq!(Activation::ReLU.apply(0.0), 0.0);
    }

    #[test]
    fn relu_derivative_positive() {
        assert_eq!(Activation::ReLU.derivative(5.0), 1.0);
    }

    #[test]
    fn relu_derivative_negative() {
        assert_eq!(Activation::ReLU.derivative(-5.0), 0.0);
    }

    // ── GeLU tests ───────────────────────────────────────────

    #[test]
    fn gelu_zero() {
        assert!(approx_eq(Activation::GeLU.apply(0.0), 0.0, EPS));
    }

    #[test]
    fn gelu_positive() {
        // GeLU(1.0) ≈ 0.8413
        let v = Activation::GeLU.apply(1.0);
        assert!(approx_eq(v, 0.8413, 1e-3));
    }

    #[test]
    fn gelu_negative() {
        // GeLU(-1.0) ≈ -0.1587
        let v = Activation::GeLU.apply(-1.0);
        assert!(approx_eq(v, -0.1587, 1e-3));
    }

    #[test]
    fn gelu_large_positive() {
        let v = Activation::GeLU.apply(10.0);
        assert!(approx_eq(v, 10.0, 1e-3));
    }

    // ── GeLU Approx tests ────────────────────────────────────

    #[test]
    fn gelu_approx_close_to_exact() {
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let exact = Activation::GeLU.apply(x);
            let approx = Activation::GeLUApprox.apply(x);
            assert!(approx_eq(exact, approx, 0.02), "GeLU vs approx at x={x}: {exact} vs {approx}");
        }
    }

    #[test]
    fn gelu_approx_zero() {
        assert!(approx_eq(Activation::GeLUApprox.apply(0.0), 0.0, EPS));
    }

    // ── SiLU tests ───────────────────────────────────────────

    #[test]
    fn silu_zero() {
        assert!(approx_eq(Activation::SiLU.apply(0.0), 0.0, EPS));
    }

    #[test]
    fn silu_positive() {
        // SiLU(1.0) = 1 * sigmoid(1) ≈ 0.7311
        let v = Activation::SiLU.apply(1.0);
        assert!(approx_eq(v, 0.7311, 1e-3));
    }

    #[test]
    fn silu_negative() {
        // SiLU(-1.0) = -1 * sigmoid(-1) ≈ -0.2689
        let v = Activation::SiLU.apply(-1.0);
        assert!(approx_eq(v, -0.2689, 1e-3));
    }

    // ── Sigmoid tests ────────────────────────────────────────

    #[test]
    fn sigmoid_zero() {
        assert!(approx_eq(Activation::Sigmoid.apply(0.0), 0.5, EPS));
    }

    #[test]
    fn sigmoid_large_positive() {
        assert!(approx_eq(Activation::Sigmoid.apply(100.0), 1.0, EPS));
    }

    #[test]
    fn sigmoid_large_negative() {
        assert!(approx_eq(Activation::Sigmoid.apply(-100.0), 0.0, EPS));
    }

    #[test]
    fn sigmoid_one() {
        let v = Activation::Sigmoid.apply(1.0);
        assert!(approx_eq(v, 0.7311, 1e-3));
    }

    #[test]
    fn sigmoid_derivative_at_zero() {
        // sigmoid'(0) = 0.25
        assert!(approx_eq(Activation::Sigmoid.derivative(0.0), 0.25, EPS));
    }

    // ── Tanh tests ───────────────────────────────────────────

    #[test]
    fn tanh_zero() {
        assert!(approx_eq(Activation::Tanh.apply(0.0), 0.0, EPS));
    }

    #[test]
    fn tanh_positive() {
        assert!(approx_eq(Activation::Tanh.apply(1.0), 1.0_f32.tanh(), EPS));
    }

    #[test]
    fn tanh_large_positive() {
        assert!(approx_eq(Activation::Tanh.apply(100.0), 1.0, EPS));
    }

    #[test]
    fn tanh_derivative_at_zero() {
        assert!(approx_eq(Activation::Tanh.derivative(0.0), 1.0, EPS));
    }

    // ── Softplus tests ───────────────────────────────────────

    #[test]
    fn softplus_zero() {
        // softplus(0) = ln(2) ≈ 0.6931
        assert!(approx_eq(Activation::Softplus.apply(0.0), 2.0_f32.ln(), EPS));
    }

    #[test]
    fn softplus_large() {
        // For large x, softplus(x) ≈ x
        assert!(approx_eq(Activation::Softplus.apply(50.0), 50.0, EPS));
    }

    #[test]
    fn softplus_negative() {
        // softplus(-50) ≈ 0
        assert!(approx_eq(Activation::Softplus.apply(-50.0), 0.0, EPS));
    }

    #[test]
    fn softplus_derivative_is_sigmoid() {
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let d = Activation::Softplus.derivative(x);
            let s = Activation::Sigmoid.apply(x);
            assert!(approx_eq(d, s, EPS), "softplus'({x}) = {d}, sigmoid({x}) = {s}");
        }
    }

    // ── Mish tests ───────────────────────────────────────────

    #[test]
    fn mish_zero() {
        assert!(approx_eq(Activation::Mish.apply(0.0), 0.0, 1e-3));
    }

    #[test]
    fn mish_positive() {
        // Mish(1.0) = 1.0 * tanh(softplus(1.0)) ≈ 0.8651
        let v = Activation::Mish.apply(1.0);
        assert!(approx_eq(v, 0.8651, 1e-3));
    }

    #[test]
    fn mish_negative() {
        let v = Activation::Mish.apply(-1.0);
        // Mish(-1) = -1 * tanh(softplus(-1)) ≈ -0.3034
        assert!(approx_eq(v, -0.3034, 1e-3));
    }

    // ── Numerical derivative checks ─────────────────────────

    #[test]
    fn numerical_derivative_gelu() {
        check_numerical_derivative(Activation::GeLU);
    }

    #[test]
    fn numerical_derivative_gelu_approx() {
        check_numerical_derivative(Activation::GeLUApprox);
    }

    #[test]
    fn numerical_derivative_silu() {
        check_numerical_derivative(Activation::SiLU);
    }

    #[test]
    fn numerical_derivative_sigmoid() {
        check_numerical_derivative(Activation::Sigmoid);
    }

    #[test]
    fn numerical_derivative_tanh() {
        check_numerical_derivative(Activation::Tanh);
    }

    #[test]
    fn numerical_derivative_softplus() {
        check_numerical_derivative(Activation::Softplus);
    }

    #[test]
    fn numerical_derivative_mish() {
        check_numerical_derivative(Activation::Mish);
    }

    fn check_numerical_derivative(act: Activation) {
        let h = 1e-4_f32;
        for &x in &[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] {
            let numerical = (act.apply(x + h) - act.apply(x - h)) / (2.0 * h);
            let analytic = act.derivative(x);
            assert!(
                approx_eq(numerical, analytic, 1e-2),
                "{}: d/dx at x={x}: numerical={numerical}, \
                 analytic={analytic}",
                act.name()
            );
        }
    }

    // ── In-place vs batch consistency ────────────────────────

    #[test]
    fn inplace_vs_batch_all_activations() {
        let input = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        for &act in ALL_ACTIVATIONS {
            let mut inplace = input.clone();
            act.apply_inplace(&mut inplace);

            let mut batch = vec![0.0; input.len()];
            act.apply_batch(&input, &mut batch);

            for i in 0..input.len() {
                assert!(
                    approx_eq(inplace[i], batch[i], 1e-7),
                    "{}: mismatch at index {i}",
                    act.name()
                );
            }
        }
    }

    // ── Special values ───────────────────────────────────────

    #[test]
    fn special_value_nan() {
        // ReLU uses f32::max which does not propagate NaN (IEEE 754).
        for &act in ALL_ACTIVATIONS {
            if act == Activation::ReLU {
                continue;
            }
            let v = act.apply(f32::NAN);
            assert!(v.is_nan(), "{} should return NaN for NaN", act.name());
        }
    }

    #[test]
    fn relu_nan_returns_zero() {
        // f32::max(NaN, 0.0) == 0.0 per IEEE 754
        assert_eq!(Activation::ReLU.apply(f32::NAN), 0.0);
    }

    #[test]
    fn special_value_positive_inf() {
        assert_eq!(Activation::ReLU.apply(f32::INFINITY), f32::INFINITY);
        assert_eq!(Activation::Sigmoid.apply(f32::INFINITY), 1.0);
        assert_eq!(Activation::Tanh.apply(f32::INFINITY), 1.0);
    }

    #[test]
    fn special_value_negative_inf() {
        assert_eq!(Activation::ReLU.apply(f32::NEG_INFINITY), 0.0);
        assert_eq!(Activation::Sigmoid.apply(f32::NEG_INFINITY), 0.0);
        assert_eq!(Activation::Tanh.apply(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn special_value_one() {
        for &act in ALL_ACTIVATIONS {
            let v = act.apply(1.0);
            assert!(v.is_finite(), "{} at 1.0 not finite", act.name());
        }
    }

    #[test]
    fn special_value_neg_one() {
        for &act in ALL_ACTIVATIONS {
            let v = act.apply(-1.0);
            assert!(v.is_finite(), "{} at -1.0 not finite", act.name());
        }
    }

    #[test]
    fn special_value_very_large() {
        for &act in ALL_ACTIVATIONS {
            let v = act.apply(1e6);
            assert!(!v.is_nan(), "{} at 1e6 is NaN", act.name());
        }
    }

    #[test]
    fn special_value_very_small() {
        for &act in ALL_ACTIVATIONS {
            let v = act.apply(-1e6);
            assert!(!v.is_nan(), "{} at -1e6 is NaN", act.name());
        }
    }

    // ── Fused add + activation ───────────────────────────────

    #[test]
    fn fused_add_relu() {
        let input = vec![1.0, -2.0, 3.0];
        let residual = vec![0.5, 3.0, -4.0];
        let mut output = vec![0.0; 3];
        fused_add_activation(&input, &residual, &mut output, Activation::ReLU);
        assert_eq!(output, vec![1.5, 1.0, 0.0]);
    }

    #[test]
    fn fused_add_sigmoid() {
        let input = vec![0.0];
        let residual = vec![0.0];
        let mut output = vec![0.0];
        fused_add_activation(&input, &residual, &mut output, Activation::Sigmoid);
        assert!(approx_eq(output[0], 0.5, EPS));
    }

    #[test]
    fn fused_add_matches_separate() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let residual = vec![0.2, 0.3, -0.1, 0.8];
        for &act in ALL_ACTIVATIONS {
            let mut fused = vec![0.0; input.len()];
            fused_add_activation(&input, &residual, &mut fused, act);
            let combined: Vec<f32> =
                input.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
            let mut separate = vec![0.0; input.len()];
            act.apply_batch(&combined, &mut separate);
            for i in 0..input.len() {
                assert!(
                    approx_eq(fused[i], separate[i], 1e-7),
                    "{}: fused mismatch at {i}",
                    act.name()
                );
            }
        }
    }

    // ── OpenCL source strings ────────────────────────────────

    #[test]
    fn opencl_sources_non_empty() {
        for &act in ALL_ACTIVATIONS {
            let src = act.opencl_source();
            assert!(!src.is_empty(), "{} OpenCL source is empty", act.name());
        }
    }

    #[test]
    fn opencl_sources_contain_kernel() {
        for &act in ALL_ACTIVATIONS {
            let src = act.opencl_source();
            assert!(src.contains("__kernel"), "{} OpenCL source missing __kernel", act.name());
        }
    }

    // ── WGSL source strings ─────────────────────────────────

    #[test]
    fn wgsl_sources_non_empty() {
        for &act in ALL_ACTIVATIONS {
            let src = act.wgsl_source();
            assert!(!src.is_empty(), "{} WGSL source is empty", act.name());
        }
    }

    #[test]
    fn wgsl_sources_contain_fn() {
        for &act in ALL_ACTIVATIONS {
            let src = act.wgsl_source();
            assert!(src.contains("fn "), "{} WGSL source missing fn keyword", act.name());
        }
    }

    // ── Activation names ─────────────────────────────────────

    #[test]
    fn all_names_non_empty() {
        for &act in ALL_ACTIVATIONS {
            assert!(!act.name().is_empty());
        }
    }

    #[test]
    fn all_variants_in_constant() {
        assert_eq!(ALL_ACTIVATIONS.len(), 8);
    }

    // ── Monotonicity checks ──────────────────────────────────

    #[test]
    fn relu_monotonic() {
        check_monotonic(Activation::ReLU);
    }

    #[test]
    fn sigmoid_monotonic() {
        check_monotonic(Activation::Sigmoid);
    }

    #[test]
    fn tanh_monotonic() {
        check_monotonic(Activation::Tanh);
    }

    #[test]
    fn softplus_monotonic() {
        check_monotonic(Activation::Softplus);
    }

    fn check_monotonic(act: Activation) {
        let mut prev = act.apply(-10.0);
        let mut x = -9.9_f32;
        while x <= 10.0 {
            let cur = act.apply(x);
            assert!(cur >= prev - EPS, "{} not monotonic at x={x}: {prev} > {cur}", act.name());
            prev = cur;
            x += 0.1;
        }
    }

    // ── Batch panics on mismatched lengths ───────────────────

    #[test]
    #[should_panic(expected = "equal length")]
    fn batch_panics_on_length_mismatch() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0];
        Activation::ReLU.apply_batch(&input, &mut output);
    }

    #[test]
    #[should_panic]
    fn fused_panics_on_length_mismatch() {
        let input = vec![1.0];
        let residual = vec![1.0, 2.0];
        let mut output = vec![0.0];
        fused_add_activation(&input, &residual, &mut output, Activation::ReLU);
    }

    // ── Empty input ──────────────────────────────────────────

    #[test]
    fn empty_inplace() {
        let mut data: Vec<f32> = vec![];
        Activation::ReLU.apply_inplace(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn empty_batch() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        Activation::ReLU.apply_batch(&input, &mut output);
        assert!(output.is_empty());
    }

    // ── Debug / Clone / Copy / Eq / Hash ─────────────────────

    #[test]
    fn activation_debug() {
        let s = format!("{:?}", Activation::ReLU);
        assert!(s.contains("ReLU"));
    }

    #[test]
    fn activation_clone_eq() {
        let a = Activation::GeLU;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn activation_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for &act in ALL_ACTIVATIONS {
            set.insert(act);
        }
        assert_eq!(set.len(), 8);
    }
}
